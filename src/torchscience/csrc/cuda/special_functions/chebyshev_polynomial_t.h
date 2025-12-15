#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::cuda::special_functions {

at::Tensor chebyshev_polynomial_t(const at::Tensor& v, const at::Tensor& z);

// Fused backward - returns (grad_v, grad_z)
std::tuple<at::Tensor, at::Tensor> chebyshev_polynomial_t_backward(
    const at::Tensor& grad_output,
    const at::Tensor& v,
    const at::Tensor& z,
    bool v_requires_grad
);

// Fused double-backward - returns (grad_grad_output, grad_v, grad_z)
std::tuple<at::Tensor, at::Tensor, at::Tensor> chebyshev_polynomial_t_backward_backward(
    const at::Tensor& ggv,
    const at::Tensor& ggz,
    const at::Tensor& grad_output,
    const at::Tensor& v,
    const at::Tensor& z,
    bool has_ggv,
    bool has_ggz
);

}  // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
  m.impl("chebyshev_polynomial_t", &torchscience::cuda::special_functions::chebyshev_polynomial_t);
  m.impl("chebyshev_polynomial_t_backward", &torchscience::cuda::special_functions::chebyshev_polynomial_t_backward);
  m.impl("chebyshev_polynomial_t_backward_backward", &torchscience::cuda::special_functions::chebyshev_polynomial_t_backward_backward);
}
