#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

namespace torchscience::meta {

inline at::Tensor hypergeometric_2_f_1_forward(
  const at::Tensor &a,
  const at::Tensor &b,
  const at::Tensor &c,
  const at::Tensor &z
) {
  at::Tensor output;

  return at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(a)
    .add_const_input(b)
    .add_const_input(c)
    .add_const_input(z)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build()
    .output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hypergeometric_2_f_1_backward(
  const at::Tensor &grad,
  const at::Tensor &a,
  const at::Tensor &b,
  const at::Tensor &c,
  const at::Tensor &z
) {
  at::Tensor o1;
  at::Tensor o2;
  at::Tensor o3;
  at::Tensor o4;

  auto iterator = at::TensorIteratorConfig()
    .add_output(o1)
    .add_output(o2)
    .add_output(o3)
    .add_output(o4)
    .add_const_input(grad)
    .add_const_input(a)
    .add_const_input(b)
    .add_const_input(c)
    .add_const_input(z)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  return {
    iterator.output(0),
    iterator.output(1),
    iterator.output(2),
    iterator.output(3)
  };
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
  if (!gg_a.defined() && !gg_b.defined() && !gg_c.defined() && !gg_z.defined()) {
    return {};
  }

  at::Tensor gg_a_safe;
  at::Tensor gg_b_safe;
  at::Tensor gg_c_safe;
  at::Tensor gg_z_safe;

  if (gg_a.defined()) {
    gg_a_safe = gg_a;
  } else {
    gg_a_safe = zeros_like(grad);
  }

  if (gg_b.defined()) {
    gg_b_safe = gg_b;
  } else {
    gg_b_safe = zeros_like(grad);
  }

  if (gg_c.defined()) {
    gg_c_safe = gg_c;
  } else {
    gg_c_safe = zeros_like(grad);
  }

  if (gg_z.defined()) {
    gg_z_safe = gg_z;
  } else {
    gg_z_safe = zeros_like(grad);
  }

  at::Tensor o1;
  at::Tensor o2;
  at::Tensor o3;
  at::Tensor o4;
  at::Tensor o5;

  auto iterator = at::TensorIteratorConfig()
    .add_output(o1)
    .add_output(o2)
    .add_output(o3)
    .add_output(o4)
    .add_output(o5)
    .add_const_input(gg_a_safe)
    .add_const_input(gg_b_safe)
    .add_const_input(gg_c_safe)
    .add_const_input(gg_z_safe)
    .add_const_input(grad)
    .add_const_input(a)
    .add_const_input(b)
    .add_const_input(c)
    .add_const_input(z)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  return {
    iterator.output(0),
    iterator.output(1),
    iterator.output(2),
    iterator.output(3),
    iterator.output(4)
  };
}

} // namespace torchscience::meta

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
  module.impl(
    "hypergeometric_2_f_1",
    torchscience::meta::hypergeometric_2_f_1_forward
  );

  module.impl(
    "hypergeometric_2_f_1_backward",
    torchscience::meta::hypergeometric_2_f_1_backward
  );

  module.impl(
    "hypergeometric_2_f_1_backward_backward",
    torchscience::meta::hypergeometric_2_f_1_backward_backward
  );
}
