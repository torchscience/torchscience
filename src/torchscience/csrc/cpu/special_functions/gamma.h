#pragma once

#include <cmath>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

namespace torchscience::cpu {

namespace {

template <typename T> T gamma_forward_kernel(T z) {
  constexpr double kGammaG = 7.0;
  constexpr double kGammaCoefficients[] = {
      0.99999999999980993,  676.5203681218851,
      -1259.1392167224028,  771.32342877765313,
      -176.61502916214059,  12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6,
      1.5056327351493116e-7};

  if (z < T(0.5)) {
    return static_cast<T>(M_PI) / (std::sin(static_cast<T>(M_PI) * z) * gamma_forward_kernel(T(1) - z));
  }

  T z_adj = z - T(1);
  T x = static_cast<T>(kGammaCoefficients[0]);
  for (int i = 1; i < 9; ++i) {
    x += static_cast<T>(kGammaCoefficients[i]) / (z_adj + T(i));
  }

  const T g = static_cast<T>(kGammaG);
  T t = z_adj + g + T(0.5);
  return std::sqrt(static_cast<T>(2 * M_PI)) * std::pow(t, z_adj + T(0.5)) * std::exp(-t) * x;
}

template <typename T> T digamma_kernel(T x) {
  T result = T(0);
  while (x < T(6)) {
    result -= T(1) / x;
    x += T(1);
  }
  T x2 = T(1) / (x * x);
  result += std::log(x) - T(0.5) / x -
            x2 * (T(1.0 / 12) - x2 * (T(1.0 / 120) - x2 * T(1.0 / 252)));
  return result;
}

template <typename T> T trigamma_kernel(T x) {
  T result = T(0);
  while (x < T(6)) {
    result += T(1) / (x * x);
    x += T(1);
  }
  T x2 = T(1) / (x * x);
  result += T(1) / x + T(0.5) * x2 +
            x2 / x * (T(1.0 / 6) - x2 * (T(1.0 / 30) - x2 * T(1.0 / 42)));
  return result;
}

} // anonymous namespace

inline at::Tensor gamma_forward(
  const at::Tensor &input
) {
  at::Tensor output;

  auto iterator = at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(input)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "gamma_cpu",
    [&] {
      at::native::cpu_kernel(iterator, gamma_forward_kernel<scalar_t>);
    }
  );

  return iterator.output();
}

inline at::Tensor gamma_backward(
  const at::Tensor &grad,
  const at::Tensor &input
) {
  at::Tensor grad_input;

  auto iterator = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_const_input(grad)
    .add_const_input(input)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "gamma_backward_cpu",
    [&] {
      at::native::cpu_kernel(
        iterator,
        [](
          scalar_t g,
          scalar_t z
        ) -> scalar_t {
          return g * gamma_forward_kernel(z) * digamma_kernel(z);
        }
      );
    }
  );

  return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> gamma_backward_backward(
  const at::Tensor &gg_input,
  const at::Tensor &grad,
  const at::Tensor &input
) {
  if (!gg_input.defined()) {
    return {at::Tensor(), at::Tensor()};
  }

  at::Tensor gg_output;
  at::Tensor new_grad;

  auto iterator = at::TensorIteratorConfig()
    .add_output(gg_output)
    .add_output(new_grad)
    .add_const_input(gg_input)
    .add_const_input(grad)
    .add_const_input(input)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "gamma_backward_backward_cpu",
    [&] {
      at::native::cpu_kernel_multiple_outputs(
        iterator,
        [](
          scalar_t gg,
          scalar_t g,
          scalar_t z
        ) -> std::tuple<scalar_t, scalar_t> {
          scalar_t gamma_z = gamma_forward_kernel(z);
          scalar_t psi_z = digamma_kernel(z);
          scalar_t psi1_z = trigamma_kernel(z);
          scalar_t gg_out = gg * gamma_z * psi_z;
          scalar_t new_grad = gg * g * gamma_z * (psi_z * psi_z + psi1_z);

          return {
            gg_out,
            new_grad
          };
        }
      );
    }
  );

  return {
    iterator.output(0),
    iterator.output(1)
  };
}

} // namespace torchscience::cpu

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
  module.impl(
    "gamma",
    torchscience::cpu::gamma_forward
  );

  module.impl(
    "gamma_backward",
    torchscience::cpu::gamma_backward
  );

  module.impl(
    "gamma_backward_backward",
    torchscience::cpu::gamma_backward_backward
  );
}
