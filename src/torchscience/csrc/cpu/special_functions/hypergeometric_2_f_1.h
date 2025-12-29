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
T hyp2f1_forward_kernel(T a, T b, T c, T z) {
  // Special case: z = 0
  if (std::abs(z) < epsilon<T>()) {
    return T(1);
  }

  // For now, only handle |z| < 0.5 with direct series
  if (std::abs(z) < T(0.5)) {
    return hyp2f1_series(a, b, c, z);
  }

  // Placeholder for other regions - will be implemented in later tasks
  return std::numeric_limits<T>::quiet_NaN();
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
  TORCH_CHECK(false, "hypergeometric_2_f_1_backward not yet implemented");
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
