#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

namespace torchscience::meta {

inline at::Tensor incomplete_beta_forward(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& x
) {
    at::Tensor output;
    return at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(a)
        .add_const_input(b)
        .add_const_input(x)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build()
        .output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> incomplete_beta_backward(
    const at::Tensor& grad,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& x
) {
    at::Tensor o1, o2, o3;
    auto iter = at::TensorIteratorConfig()
        .add_output(o1)
        .add_output(o2)
        .add_output(o3)
        .add_const_input(grad)
        .add_const_input(a)
        .add_const_input(b)
        .add_const_input(x)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();
    return {iter.output(0), iter.output(1), iter.output(2)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> incomplete_beta_backward_backward(
    const at::Tensor& gg_a,
    const at::Tensor& gg_b,
    const at::Tensor& gg_x,
    const at::Tensor& grad,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& x
) {
    if (!gg_a.defined() && !gg_b.defined() && !gg_x.defined()) {
        return {};
    }
    at::Tensor gg_a_safe = gg_a.defined() ? gg_a : at::zeros_like(grad);
    at::Tensor gg_b_safe = gg_b.defined() ? gg_b : at::zeros_like(grad);
    at::Tensor gg_x_safe = gg_x.defined() ? gg_x : at::zeros_like(grad);
    at::Tensor o1, o2, o3, o4;
    auto iter = at::TensorIteratorConfig()
        .add_output(o1)
        .add_output(o2)
        .add_output(o3)
        .add_output(o4)
        .add_const_input(gg_a_safe)
        .add_const_input(gg_b_safe)
        .add_const_input(gg_x_safe)
        .add_const_input(grad)
        .add_const_input(a)
        .add_const_input(b)
        .add_const_input(x)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();
    return {iter.output(0), iter.output(1), iter.output(2), iter.output(3)};
}

}  // namespace torchscience::meta

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("incomplete_beta", torchscience::meta::incomplete_beta_forward);
    m.impl("incomplete_beta_backward", torchscience::meta::incomplete_beta_backward);
    m.impl("incomplete_beta_backward_backward", torchscience::meta::incomplete_beta_backward_backward);
}
