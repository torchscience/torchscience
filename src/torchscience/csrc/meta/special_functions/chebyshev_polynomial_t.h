#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

namespace torchscience::meta {

inline at::Tensor chebyshev_polynomial_t_forward(
  const at::Tensor &x,
  const at::Tensor &n
) {
  at::Tensor output;

  return at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build()
    .output();
}

inline std::tuple<at::Tensor, at::Tensor> chebyshev_polynomial_t_backward(
  const at::Tensor &grad,
  const at::Tensor &x,
  const at::Tensor &n
) {
  at::Tensor o1;
  at::Tensor o2;

  auto iterator = at::TensorIteratorConfig()
    .add_output(o1)
    .add_output(o2)
    .add_const_input(grad)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  return {
    iterator.output(0),
    iterator.output(1)
  };
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> chebyshev_polynomial_t_backward_backward(
  const at::Tensor &gg_x,
  const at::Tensor &gg_n,
  const at::Tensor &grad,
  const at::Tensor &x,
  const at::Tensor &n
) {
  if (!gg_x.defined() && !gg_n.defined()) {
    return {};
  }

  at::Tensor gg_x_safe = gg_x.defined() ? gg_x : at::zeros_like(grad);
  at::Tensor gg_n_safe = gg_n.defined() ? gg_n : at::zeros_like(grad);

  at::Tensor o1;
  at::Tensor o2;
  at::Tensor o3;

  auto iterator = at::TensorIteratorConfig()
    .add_output(o1)
    .add_output(o2)
    .add_output(o3)
    .add_const_input(gg_x_safe)
    .add_const_input(gg_n_safe)
    .add_const_input(grad)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  return {
    iterator.output(0),
    iterator.output(1),
    iterator.output(2)
  };
}

} // namespace torchscience::meta

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
  module.impl(
    "chebyshev_polynomial_t",
    torchscience::meta::chebyshev_polynomial_t_forward
  );

  module.impl(
    "chebyshev_polynomial_t_backward",
    torchscience::meta::chebyshev_polynomial_t_backward
  );

  module.impl(
    "chebyshev_polynomial_t_backward_backward",
    torchscience::meta::chebyshev_polynomial_t_backward_backward
  );
}
