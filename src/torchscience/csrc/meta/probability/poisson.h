#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor poisson_cdf(
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto result_shape = at::infer_size(k.sizes(), rate.sizes());
  return at::empty(result_shape, k.options());
}

std::tuple<at::Tensor, at::Tensor> poisson_cdf_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& rate) {
  return std::make_tuple(
      at::empty_like(k),
      at::empty_like(rate));
}

at::Tensor poisson_pmf(
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto result_shape = at::infer_size(k.sizes(), rate.sizes());
  return at::empty(result_shape, k.options());
}

std::tuple<at::Tensor, at::Tensor> poisson_pmf_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& rate) {
  return std::make_tuple(
      at::empty_like(k),
      at::empty_like(rate));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("poisson_cdf", &poisson_cdf);
  m.impl("poisson_cdf_backward", &poisson_cdf_backward);
  m.impl("poisson_pmf", &poisson_pmf);
  m.impl("poisson_pmf_backward", &poisson_pmf_backward);
}

}  // namespace torchscience::meta::probability
