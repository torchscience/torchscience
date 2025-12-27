#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

namespace torchscience::meta {

inline at::Tensor gamma_forward(const at::Tensor& input) {
    at::Tensor output;
    return at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build()
        .output();
}

inline at::Tensor gamma_backward(const at::Tensor& grad, const at::Tensor& input) {
    at::Tensor grad_input;
    return at::TensorIteratorConfig()
        .add_output(grad_input)
        .add_const_input(grad)
        .add_const_input(input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build()
        .output();
}

inline std::tuple<at::Tensor, at::Tensor> gamma_backward_backward(
    const at::Tensor& gg,
    const at::Tensor& grad,
    const at::Tensor& input
) {
    if (!gg.defined()) {
        return {};
    }
    at::Tensor o1, o2;
    auto iter = at::TensorIteratorConfig()
        .add_output(o1)
        .add_output(o2)
        .add_const_input(gg)
        .add_const_input(grad)
        .add_const_input(input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();
    return {iter.output(0), iter.output(1)};
}

}  // namespace torchscience::meta

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("gamma", torchscience::meta::gamma_forward);
    m.impl("gamma_backward", torchscience::meta::gamma_backward);
    m.impl("gamma_backward_backward", torchscience::meta::gamma_backward_backward);
}
