#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

#include "../../kernel/special_functions/incomplete_beta.h"
#include "../../kernel/special_functions/incomplete_beta_backward.h"
#include "../../kernel/special_functions/incomplete_beta_backward_backward.h"

namespace torchscience::cpu {

inline at::Tensor incomplete_beta_forward(
  const at::Tensor &x,
  const at::Tensor &a,
  const at::Tensor &b
) {
  at::Tensor output;

  auto iterator = at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(x)
    .add_const_input(a)
    .add_const_input(b)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "incomplete_beta_cpu",
    [&] {
      at::native::cpu_kernel(
        iterator,
        [](scalar_t x_val, scalar_t a_val, scalar_t b_val) -> scalar_t {
          return kernel::special_functions::incomplete_beta(x_val, a_val, b_val);
        }
      );
    }
  );

  return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> incomplete_beta_backward(
  const at::Tensor &grad,
  const at::Tensor &x,
  const at::Tensor &a,
  const at::Tensor &b
) {
  at::Tensor grad_x;
  at::Tensor grad_a;
  at::Tensor grad_b;

  auto iterator = at::TensorIteratorConfig()
    .add_output(grad_x)
    .add_output(grad_a)
    .add_output(grad_b)
    .add_const_input(grad)
    .add_const_input(x)
    .add_const_input(a)
    .add_const_input(b)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "incomplete_beta_backward_cpu",
    [&] {
      at::native::cpu_kernel_multiple_outputs(
        iterator,
        [](scalar_t g, scalar_t x_val, scalar_t a_val, scalar_t b_val)
            -> std::tuple<scalar_t, scalar_t, scalar_t> {
          return kernel::special_functions::incomplete_beta_backward(g, x_val, a_val, b_val);
        }
      );
    }
  );

  return {iterator.output(0), iterator.output(1), iterator.output(2)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> incomplete_beta_backward_backward(
  const at::Tensor &gg_x,
  const at::Tensor &gg_a,
  const at::Tensor &gg_b,
  const at::Tensor &grad,
  const at::Tensor &x,
  const at::Tensor &a,
  const at::Tensor &b
) {
  bool has_gg_x = gg_x.defined();
  bool has_gg_a = gg_a.defined();
  bool has_gg_b = gg_b.defined();

  if (!has_gg_x && !has_gg_a && !has_gg_b) {
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }

  at::Tensor gg_x_safe = has_gg_x ? gg_x : at::zeros_like(grad);
  at::Tensor gg_a_safe = has_gg_a ? gg_a : at::zeros_like(grad);
  at::Tensor gg_b_safe = has_gg_b ? gg_b : at::zeros_like(grad);

  at::Tensor gg_out;
  at::Tensor new_grad_x;
  at::Tensor new_grad_a;
  at::Tensor new_grad_b;

  auto iterator = at::TensorIteratorConfig()
    .add_output(gg_out)
    .add_output(new_grad_x)
    .add_output(new_grad_a)
    .add_output(new_grad_b)
    .add_const_input(gg_x_safe)
    .add_const_input(gg_a_safe)
    .add_const_input(gg_b_safe)
    .add_const_input(grad)
    .add_const_input(x)
    .add_const_input(a)
    .add_const_input(b)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "incomplete_beta_backward_backward_cpu",
    [&] {
      at::native::cpu_kernel_multiple_outputs(
        iterator,
        [has_gg_x, has_gg_a, has_gg_b](
            scalar_t gg_x_val, scalar_t gg_a_val, scalar_t gg_b_val,
            scalar_t g, scalar_t x_val, scalar_t a_val, scalar_t b_val)
            -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
          return kernel::special_functions::incomplete_beta_backward_backward(
            gg_x_val, gg_a_val, gg_b_val,
            g, x_val, a_val, b_val,
            has_gg_x, has_gg_a, has_gg_b
          );
        }
      );
    }
  );

  return {
    iterator.output(0),
    iterator.output(1),
    iterator.output(2),
    iterator.output(3)
  };
}

} // namespace torchscience::cpu

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
  module.impl("incomplete_beta", torchscience::cpu::incomplete_beta_forward);

  module.impl("incomplete_beta_backward", torchscience::cpu::incomplete_beta_backward);

  module.impl("incomplete_beta_backward_backward", torchscience::cpu::incomplete_beta_backward_backward);
}
