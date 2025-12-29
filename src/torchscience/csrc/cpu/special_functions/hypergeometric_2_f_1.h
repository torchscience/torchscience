#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <tuple>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/complex.h>
#include <torch/library.h>

namespace torchscience::cpu {

namespace {

// Helper to check if type is complex (either std::complex or c10::complex)
template <typename T>
struct is_complex_type : std::false_type {};

template <typename T>
struct is_complex_type<std::complex<T>> : std::true_type {};

template <typename T>
struct is_complex_type<c10::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex_type<T>::value;

// Get the real value type from complex
template <typename T>
struct real_type { using type = T; };

template <typename T>
struct real_type<std::complex<T>> { using type = T; };

template <typename T>
struct real_type<c10::complex<T>> { using type = T; };

template <typename T>
using real_type_t = typename real_type<T>::type;

template <typename T>
constexpr auto epsilon() {
  using real_t = real_type_t<T>;
  if constexpr (std::is_same_v<real_t, float>) {
    return float(1e-7);
  } else if constexpr (std::is_same_v<real_t, double>) {
    return double(1e-15);
  } else {
    // For half types, use float epsilon
    return float(1e-7);
  }
}

template <typename T>
bool is_nonpositive_integer(T x) {
  if constexpr (is_complex_v<T>) {
    using real_t = real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < epsilon<T>() &&
           re <= real_t(0) &&
           std::abs(re - std::round(re)) < epsilon<T>();
  } else {
    // Convert to double for comparison to handle BFloat16/Half
    double xd = static_cast<double>(x);
    return xd <= 0.0 && std::abs(xd - std::round(xd)) < epsilon<T>();
  }
}

template <typename T>
int get_nonpositive_int(T x) {
  if constexpr (is_complex_v<T>) {
    using real_t = real_type_t<T>;
    return static_cast<int>(std::round(static_cast<real_t>(x.real())));
  } else {
    // Convert to double to handle BFloat16/Half
    return static_cast<int>(std::round(static_cast<double>(x)));
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

  // Use double for comparisons to handle BFloat16/Half
  double zt_abs;
  if constexpr (is_complex_v<T>) {
    zt_abs = std::abs(z_transformed);
  } else {
    zt_abs = std::abs(static_cast<double>(z_transformed));
  }

  if (zt_abs < 0.5) {
    return std::pow(T(1) - z, -a) * hyp2f1_series(a, c - b, c, z_transformed);
  }

  // Alternative Pfaff: 2F1(a,b;c;z) = (1-z)^(-b) * 2F1(b, c-a; c; z/(z-1))
  if (zt_abs < 0.9) {
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

  // Use double for comparisons to handle BFloat16/Half
  double w_abs = std::abs(static_cast<double>(w));

  // If |w| < 0.5, use direct series
  if (w_abs < 0.5) {
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

  // For complex types, check if z is real and negative
  // For real types, use direct comparison
  if constexpr (is_complex_v<T>) {
    using real_t = real_type_t<T>;
    // Complex case: use series directly for |z| < 0.5, transformations otherwise
    if (std::abs(z) < real_t(0.5)) {
      return hyp2f1_series(a, b, c, z);
    }

    // For |z| in [0.5, 1), use Pfaff transformation
    if (std::abs(z) < real_t(1)) {
      return hyp2f1_near_one(a, b, c, z);
    }

    // For |z| >= 1, use Pfaff transformation with more iterations
    // This handles the analytic continuation for complex z
    T w = z / (z - T(1));
    return std::pow(T(1) - z, -a) * hyp2f1_series(a, c - b, c, w, 2000);
  } else {
    // Real case - use double for comparisons to handle BFloat16/Half
    double zd = static_cast<double>(z);

    // Negative z: use Pfaff transformation z -> z/(z-1)
    if (zd < 0.0) {
      return hyp2f1_negative_z(a, b, c, z);
    }

    // Direct series for |z| < 0.5
    if (std::abs(zd) < 0.5) {
      return hyp2f1_series(a, b, c, z);
    }

    // z in [0.5, 1): use Pfaff transformation
    if (std::abs(zd) < 1.0) {
      return hyp2f1_near_one(a, b, c, z);
    }

    // z >= 1: divergent for real z on branch cut, return NaN
    return std::numeric_limits<T>::quiet_NaN();
  }
}

template <typename T>
std::tuple<T, T, T, T> hyp2f1_backward_kernel(T grad, T a, T b, T c, T z) {
  // d/dz 2F1(a,b;c;z) = (a*b/c) * 2F1(a+1, b+1; c+1; z)
  T dz = grad * (a * b / c) * hyp2f1_forward_kernel(a + T(1), b + T(1), c + T(1), z);

  // Use finite differences for parameter gradients
  // This is robust for all regions, though less efficient than analytical gradients
  T eps = std::sqrt(epsilon<T>());
  T f_center = hyp2f1_forward_kernel(a, b, c, z);

  // d/da using central difference
  T f_a_plus = hyp2f1_forward_kernel(a + eps, b, c, z);
  T f_a_minus = hyp2f1_forward_kernel(a - eps, b, c, z);
  T da = grad * (f_a_plus - f_a_minus) / (T(2) * eps);

  // d/db using central difference
  T f_b_plus = hyp2f1_forward_kernel(a, b + eps, c, z);
  T f_b_minus = hyp2f1_forward_kernel(a, b - eps, c, z);
  T db = grad * (f_b_plus - f_b_minus) / (T(2) * eps);

  // d/dc using central difference
  T f_c_plus = hyp2f1_forward_kernel(a, b, c + eps, z);
  T f_c_minus = hyp2f1_forward_kernel(a, b, c - eps, z);
  T dc = grad * (f_c_plus - f_c_minus) / (T(2) * eps);

  return {da, db, dc, dz};
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

  if (at::isComplexType(iterator.common_dtype())) {
    AT_DISPATCH_COMPLEX_TYPES(
      iterator.common_dtype(),
      "hypergeometric_2_f_1_cpu_complex",
      [&] {
        at::native::cpu_kernel(iterator, [](scalar_t a, scalar_t b, scalar_t c, scalar_t z) {
          return hyp2f1_forward_kernel(a, b, c, z);
        });
      }
    );
  } else {
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
  }

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

  if (at::isComplexType(iterator.common_dtype())) {
    AT_DISPATCH_COMPLEX_TYPES(
      iterator.common_dtype(),
      "hypergeometric_2_f_1_backward_cpu_complex",
      [&] {
        at::native::cpu_kernel_multiple_outputs(iterator, [](scalar_t grad, scalar_t a, scalar_t b, scalar_t c, scalar_t z) {
          return hyp2f1_backward_kernel(grad, a, b, c, z);
        });
      }
    );
  } else {
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
  }

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
  // Handle undefined gradients
  auto gg_a_safe = gg_a.defined() ? gg_a : at::zeros_like(grad);
  auto gg_b_safe = gg_b.defined() ? gg_b : at::zeros_like(grad);
  auto gg_c_safe = gg_c.defined() ? gg_c : at::zeros_like(grad);
  auto gg_z_safe = gg_z.defined() ? gg_z : at::zeros_like(grad);

  // Compute second derivative w.r.t. z:
  // d²/dz² 2F1(a,b;c;z) = (a*b/c) * ((a+1)*(b+1)/(c+1)) * 2F1(a+2, b+2; c+2; z)
  auto dz_coef = a * b / c;
  auto d2z_coef = dz_coef * (a + 1) * (b + 1) / (c + 1);

  // f_shifted = 2F1(a+1, b+1; c+1; z)
  auto a_plus_1 = a + 1;
  auto b_plus_1 = b + 1;
  auto c_plus_1 = c + 1;
  auto f_shifted = at::zeros_like(grad);

  // f_double_shifted = 2F1(a+2, b+2; c+2; z)
  auto a_plus_2 = a + 2;
  auto b_plus_2 = b + 2;
  auto c_plus_2 = c + 2;
  auto f_double_shifted = at::zeros_like(grad);

  // Compute using the forward function
  {
    at::AutoDispatchBelowAutograd guard;
    static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::hypergeometric_2_f_1", "")
      .typed<at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    f_shifted = op.call(a_plus_1, b_plus_1, c_plus_1, z);
    f_double_shifted = op.call(a_plus_2, b_plus_2, c_plus_2, z);
  }

  // gg_out: gradient of backward output w.r.t. upstream grad
  // backward returns (grad_a, grad_b, grad_c, grad_z) where grad_z = grad * dz_coef * f_shifted
  // So d(grad_z)/d(grad) = dz_coef * f_shifted
  auto gg_out = gg_z_safe * dz_coef * f_shifted;

  // new_grad_z: second derivative w.r.t. z
  // grad_z = grad * dz_coef * f_shifted
  // d(grad_z)/dz = grad * dz_coef * d(f_shifted)/dz
  //              = grad * dz_coef * ((a+1)*(b+1)/(c+1)) * f_double_shifted
  //              = grad * d2z_coef * f_double_shifted
  auto new_grad_z = gg_z_safe * grad * d2z_coef * f_double_shifted;

  // For parameter second derivatives, use finite differences (simplified)
  // Return zeros for now - full implementation would require more complex differentiation
  auto new_grad_a = at::zeros_like(grad);
  auto new_grad_b = at::zeros_like(grad);
  auto new_grad_c = at::zeros_like(grad);

  return {gg_out, new_grad_a, new_grad_b, new_grad_c, new_grad_z};
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
