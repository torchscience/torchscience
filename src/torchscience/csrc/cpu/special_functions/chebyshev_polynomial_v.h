#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

#include "../../kernel/special_functions/chebyshev_polynomial_v.h"
#include "../../kernel/special_functions/chebyshev_polynomial_v_backward.h"
#include "../../kernel/special_functions/chebyshev_polynomial_v_backward_backward.h"

namespace torchscience::cpu {

inline at::Tensor chebyshev_polynomial_v_forward(
  const at::Tensor &x,
  const at::Tensor &n
) {
  at::Tensor output;

  auto iterator = at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "chebyshev_polynomial_v_cpu",
    [&] {
      at::native::cpu_kernel(
        iterator,
        [](
          scalar_t x,
          scalar_t n
        ) -> scalar_t {
          return kernel::special_functions::chebyshev_polynomial_v(
            x,
            n
          );
        }
      );
    }
  );

  return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> chebyshev_polynomial_v_backward(
  const at::Tensor &grad,
  const at::Tensor &x,
  const at::Tensor &n
) {
  at::Tensor grad_x;
  at::Tensor grad_n;

  auto iterator = at::TensorIteratorConfig()
    .add_output(grad_x)
    .add_output(grad_n)
    .add_const_input(grad)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "chebyshev_polynomial_v_backward_cpu",
    [&] {
      at::native::cpu_kernel_multiple_outputs(
        iterator,
        [](scalar_t g, scalar_t x, scalar_t n) -> std::tuple<scalar_t, scalar_t> {
          return kernel::special_functions::chebyshev_polynomial_v_backward(g, x, n);
        }
      );
    }
  );

  return {iterator.output(0), iterator.output(1)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> chebyshev_polynomial_v_backward_backward(
  const at::Tensor &gg_x,
  const at::Tensor &gg_n,
  const at::Tensor &grad,
  const at::Tensor &x,
  const at::Tensor &n
) {
  bool has_gg_x = gg_x.defined();
  bool has_gg_n = gg_n.defined();

  if (!has_gg_x && !has_gg_n) {
    return {at::Tensor(), at::Tensor(), at::Tensor()};
  }

  at::Tensor gg_x_safe = has_gg_x ? gg_x : at::zeros_like(grad);
  at::Tensor gg_n_safe = has_gg_n ? gg_n : at::zeros_like(grad);

  at::Tensor gg_out;
  at::Tensor new_grad_x;
  at::Tensor new_grad_n;

  auto iterator = at::TensorIteratorConfig()
    .add_output(gg_out)
    .add_output(new_grad_x)
    .add_output(new_grad_n)
    .add_const_input(gg_x_safe)
    .add_const_input(gg_n_safe)
    .add_const_input(grad)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "chebyshev_polynomial_v_backward_backward_cpu",
    [&] {
      at::native::cpu_kernel_multiple_outputs(
        iterator,
        [has_gg_x](
          scalar_t gg_x,
          scalar_t gg_n,
          scalar_t g,
          scalar_t x,
          scalar_t n
        ) -> std::tuple<scalar_t, scalar_t, scalar_t> {
          return kernel::special_functions::chebyshev_polynomial_v_backward_backward<scalar_t>(
            gg_x,
            gg_n,
            g,
            x,
            n,
            has_gg_x
          );
        }
      );
    }
  );

  return {
    iterator.output(0),
    iterator.output(1),
    iterator.output(2)
  };
}

} // namespace torchscience::cpu

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
  module.impl(
    "chebyshev_polynomial_v",
    torchscience::cpu::chebyshev_polynomial_v_forward
  );

  module.impl(
    "chebyshev_polynomial_v_backward",
    torchscience::cpu::chebyshev_polynomial_v_backward
  );

  module.impl(
    "chebyshev_polynomial_v_backward_backward",
    torchscience::cpu::chebyshev_polynomial_v_backward_backward
  );
}
