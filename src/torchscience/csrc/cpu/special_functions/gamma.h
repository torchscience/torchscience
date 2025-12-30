#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

#include "../../kernel/special_functions/gamma.h"
#include "../../kernel/special_functions/gamma_backward.h"
#include "../../kernel/special_functions/gamma_backward_backward.h"

namespace torchscience::cpu {

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
    "gamma",
    [&] {
      at::native::cpu_kernel(
        iterator,
        [](
          scalar_t z
        ) -> scalar_t {
          return kernel::special_functions::gamma(
            z
          );
        }
      );
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
    "gamma_backward",
    [&] {
      at::native::cpu_kernel(
        iterator,
        [](
          scalar_t gradient,
          scalar_t z
        ) -> scalar_t {
          return kernel::special_functions::gamma_backward(
            gradient,
            z
          );
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
    return {
      at::Tensor(),
      at::Tensor()
    };
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
    "gamma_backward_backward",
    [&] {
      at::native::cpu_kernel_multiple_outputs(
        iterator,
        [](
          scalar_t gradient_gradient,
          scalar_t gradient,
          scalar_t z
        ) -> std::tuple<scalar_t, scalar_t> {
          return kernel::special_functions::gamma_backward_backward(
            gradient_gradient,
            gradient,
            z
          );
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
