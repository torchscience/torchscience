#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::cuda::special_functions {

at::Tensor gamma(const at::Tensor& z);

at::Tensor gamma_backward(
    const at::Tensor& grad_output,
    const at::Tensor& z
);

std::tuple<at::Tensor, at::Tensor> gamma_backward_backward(
    const at::Tensor& gg_z,
    const at::Tensor& grad_output,
    const at::Tensor& z
);

}  // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
  m.impl("gamma", &torchscience::cuda::special_functions::gamma);
  m.impl("gamma_backward", &torchscience::cuda::special_functions::gamma_backward);
  m.impl("gamma_backward_backward", &torchscience::cuda::special_functions::gamma_backward_backward);
}
