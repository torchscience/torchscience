#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

namespace torchscience::cpu {

namespace {

template <typename T>
constexpr T epsilon() {
  if constexpr (std::is_same_v<T, float>) {
    return T(1e-7);
  } else {
    return T(1e-15);
  }
}

template <typename T>
bool is_nonpositive_integer(T x) {
  if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
    return std::abs(std::imag(x)) < epsilon<typename T::value_type>() &&
           std::real(x) <= 0 &&
           std::abs(std::real(x) - std::round(std::real(x))) < epsilon<typename T::value_type>();
  } else {
    return x <= T(0) && std::abs(x - std::round(x)) < epsilon<T>();
  }
}

template <typename T>
int get_nonpositive_int(T x) {
  if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
    return static_cast<int>(std::round(std::real(x)));
  } else {
    return static_cast<int>(std::round(x));
  }
}

template <typename T>
T hyp2f1_series(T a, T b, T c, T z, int max_iter = 500) {
  T sum = T(1);
  T term = T(1);

  for (int n = 0; n < max_iter; ++n) {
    T denom = (c + T(n)) * T(n + 1);
    if (std::abs(denom) < epsilon<T>()) {
      break;
    }
    term *= (a + T(n)) * (b + T(n)) / denom * z;
    sum += term;

    if (std::abs(term) < epsilon<T>() * std::abs(sum)) {
      return sum;
    }
  }

  return sum;
}

template <typename T>
T hyp2f1_near_one(T a, T b, T c, T z) {
  // Pfaff transformation: 2F1(a,b;c;z) = (1-z)^(-a) * 2F1(a, c-b; c; z/(z-1))
  T z_transformed = z / (z - T(1));
  if (std::abs(z_transformed) < T(0.5)) {
    return std::pow(T(1) - z, -a) * hyp2f1_series(a, c - b, c, z_transformed);
  }

  // Alternative Pfaff: 2F1(a,b;c;z) = (1-z)^(-b) * 2F1(b, c-a; c; z/(z-1))
  if (std::abs(z_transformed) < T(0.9)) {
    return std::pow(T(1) - z, -b) * hyp2f1_series(b, c - a, c, z_transformed);
  }

  // Fallback: direct series with more iterations
  return hyp2f1_series(a, b, c, z, 2000);
}

template <typename T>
T hyp2f1_negative_z(T a, T b, T c, T z) {
  // For z < 0, use Pfaff transformation: z -> z/(z-1) which maps negative reals to (0,1)
  // 2F1(a,b;c;z) = (1-z)^(-a) * 2F1(a, c-b; c; z/(z-1))
  T w = z / (z - T(1));

  // w is now in (0, 1) for any z < 0
  // For z = -1: w = 0.5
  // For z -> -inf: w -> 1
  // For z -> 0-: w -> 0

  T prefactor = std::pow(T(1) - z, -a);

  // If |w| < 0.5, use direct series
  if (std::abs(w) < T(0.5)) {
    return prefactor * hyp2f1_series(a, c - b, c, w);
  }

  // For w in [0.5, 1), use more iterations
  return prefactor * hyp2f1_series(a, c - b, c, w, 1000);
}

template <typename T>
T hyp2f1_forward_kernel(T a, T b, T c, T z) {
  // Special case: z = 0
  if (std::abs(z) < epsilon<T>()) {
    return T(1);
  }

  // Special case: a = 0 or b = 0
  if (std::abs(a) < epsilon<T>() || std::abs(b) < epsilon<T>()) {
    return T(1);
  }

  // Check for pole at c = 0, -1, -2, ...
  if (is_nonpositive_integer(c)) {
    int c_int = get_nonpositive_int(c);
    // Check if pole is cancelled by a or b being "smaller" non-positive integer
    bool a_cancels = is_nonpositive_integer(a) && get_nonpositive_int(a) > c_int;
    bool b_cancels = is_nonpositive_integer(b) && get_nonpositive_int(b) > c_int;
    if (!a_cancels && !b_cancels) {
      return std::numeric_limits<T>::infinity();
    }
  }

  // Special case: c = b (reduces to power function)
  if (std::abs(c - b) < epsilon<T>()) {
    return std::pow(T(1) - z, -a);
  }

  // Special case: c = a (reduces to power function)
  if (std::abs(c - a) < epsilon<T>()) {
    return std::pow(T(1) - z, -b);
  }

  // Terminating series: a or b is non-positive integer
  if (is_nonpositive_integer(a)) {
    int n_terms = -get_nonpositive_int(a) + 1;
    return hyp2f1_series(a, b, c, z, n_terms);
  }
  if (is_nonpositive_integer(b)) {
    int n_terms = -get_nonpositive_int(b) + 1;
    return hyp2f1_series(a, b, c, z, n_terms);
  }

  // Negative z: use Pfaff transformation z -> z/(z-1)
  if (z < T(0)) {
    return hyp2f1_negative_z(a, b, c, z);
  }

  // Direct series for |z| < 0.5
  if (std::abs(z) < T(0.5)) {
    return hyp2f1_series(a, b, c, z);
  }

  // z in [0.5, 1): use Pfaff transformation
  if (std::abs(z) >= T(0.5) && std::abs(z) < T(1)) {
    return hyp2f1_near_one(a, b, c, z);
  }

  // z >= 1: divergent for real z on branch cut, return NaN
  // (Complex z > 1 will be handled in Task 8)
  return std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
std::tuple<T, T, T, T> hyp2f1_backward_kernel(T grad, T a, T b, T c, T z) {
  // d/dz 2F1(a,b;c;z) = (a*b/c) * 2F1(a+1, b+1; c+1; z)
  T dz = grad * (a * b / c) * hyp2f1_forward_kernel(a + T(1), b + T(1), c + T(1), z);

  // For now, return zeros for parameter gradients (will implement in Task 6)
  return {T(0), T(0), T(0), dz};
}

} // anonymous namespace

inline at::Tensor hypergeometric_2_f_1_forward(
  const at::Tensor &a,
  const at::Tensor &b,
  const at::Tensor &c,
  const at::Tensor &z
) {
  at::Tensor output;

  auto iterator = at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(a)
    .add_const_input(b)
    .add_const_input(c)
    .add_const_input(z)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "hypergeometric_2_f_1_cpu",
    [&] {
      at::native::cpu_kernel(iterator, [](scalar_t a, scalar_t b, scalar_t c, scalar_t z) {
        return hyp2f1_forward_kernel(a, b, c, z);
      });
    }
  );

  return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hypergeometric_2_f_1_backward(
  const at::Tensor &grad,
  const at::Tensor &a,
  const at::Tensor &b,
  const at::Tensor &c,
  const at::Tensor &z
) {
  at::Tensor grad_a, grad_b, grad_c, grad_z;

  auto iterator = at::TensorIteratorConfig()
    .add_output(grad_a)
    .add_output(grad_b)
    .add_output(grad_c)
    .add_output(grad_z)
    .add_const_input(grad)
    .add_const_input(a)
    .add_const_input(b)
    .add_const_input(c)
    .add_const_input(z)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "hypergeometric_2_f_1_backward_cpu",
    [&] {
      at::native::cpu_kernel_multiple_outputs(iterator, [](scalar_t grad, scalar_t a, scalar_t b, scalar_t c, scalar_t z) {
        return hyp2f1_backward_kernel(grad, a, b, c, z);
      });
    }
  );

  return {iterator.output(0), iterator.output(1), iterator.output(2), iterator.output(3)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> hypergeometric_2_f_1_backward_backward(
  const at::Tensor &gg_a,
  const at::Tensor &gg_b,
  const at::Tensor &gg_c,
  const at::Tensor &gg_z,
  const at::Tensor &grad,
  const at::Tensor &a,
  const at::Tensor &b,
  const at::Tensor &c,
  const at::Tensor &z
) {
  TORCH_CHECK(false, "hypergeometric_2_f_1_backward_backward not yet implemented");
}

} // namespace torchscience::cpu

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
  module.impl(
    "hypergeometric_2_f_1",
    torchscience::cpu::hypergeometric_2_f_1_forward
  );

  module.impl(
    "hypergeometric_2_f_1_backward",
    torchscience::cpu::hypergeometric_2_f_1_backward
  );

  module.impl(
    "hypergeometric_2_f_1_backward_backward",
    torchscience::cpu::hypergeometric_2_f_1_backward_backward
  );
}
