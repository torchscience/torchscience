#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

namespace torchscience::meta {

inline at::Tensor hypergeometric_2_f_1_forward(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
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
    const at::Tensor& grad,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    at::Tensor o1, o2, o3, o4;
    auto iter = at::TensorIteratorConfig()
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
    return {iter.output(0), iter.output(1), iter.output(2), iter.output(3)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> hypergeometric_2_f_1_backward_backward(
    const at::Tensor& gg_a,
    const at::Tensor& gg_b,
    const at::Tensor& gg_c,
    const at::Tensor& gg_z,
    const at::Tensor& grad,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    if (!gg_a.defined() && !gg_b.defined() && !gg_c.defined() && !gg_z.defined()) {
        return {};
    }
    at::Tensor gg_a_safe = gg_a.defined() ? gg_a : at::zeros_like(grad);
    at::Tensor gg_b_safe = gg_b.defined() ? gg_b : at::zeros_like(grad);
    at::Tensor gg_c_safe = gg_c.defined() ? gg_c : at::zeros_like(grad);
    at::Tensor gg_z_safe = gg_z.defined() ? gg_z : at::zeros_like(grad);
    at::Tensor o1, o2, o3, o4, o5;
    auto iter = at::TensorIteratorConfig()
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
    return {iter.output(0), iter.output(1), iter.output(2), iter.output(3), iter.output(4)};
}

}  // namespace torchscience::meta

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("hypergeometric_2_f_1", torchscience::meta::hypergeometric_2_f_1_forward);
    m.impl("hypergeometric_2_f_1_backward", torchscience::meta::hypergeometric_2_f_1_backward);
    m.impl("hypergeometric_2_f_1_backward_backward", torchscience::meta::hypergeometric_2_f_1_backward_backward);
}
