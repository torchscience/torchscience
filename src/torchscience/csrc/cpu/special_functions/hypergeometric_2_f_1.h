#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::cpu {

inline at::Tensor hypergeometric_2_f_1_forward(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    TORCH_CHECK(false, "hypergeometric_2_f_1 not yet implemented");
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hypergeometric_2_f_1_backward(
    const at::Tensor& grad,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    TORCH_CHECK(false, "hypergeometric_2_f_1_backward not yet implemented");
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
    TORCH_CHECK(false, "hypergeometric_2_f_1_backward_backward not yet implemented");
}

}  // namespace torchscience::cpu

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("hypergeometric_2_f_1", torchscience::cpu::hypergeometric_2_f_1_forward);
    m.impl("hypergeometric_2_f_1_backward", torchscience::cpu::hypergeometric_2_f_1_backward);
    m.impl("hypergeometric_2_f_1_backward_backward", torchscience::cpu::hypergeometric_2_f_1_backward_backward);
}
