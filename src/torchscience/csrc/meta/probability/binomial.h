#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor binomial_cdf(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto result_shape = at::infer_size(k.sizes(), n.sizes());
  result_shape = at::infer_size(result_shape, p.sizes());
  return at::empty(result_shape, k.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> binomial_cdf_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  return std::make_tuple(
      at::empty_like(k),
      at::empty_like(n),
      at::empty_like(p));
}

at::Tensor binomial_pmf(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto result_shape = at::infer_size(k.sizes(), n.sizes());
  result_shape = at::infer_size(result_shape, p.sizes());
  return at::empty(result_shape, k.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> binomial_pmf_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  return std::make_tuple(
      at::empty_like(k),
      at::empty_like(n),
      at::empty_like(p));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("binomial_cdf", &binomial_cdf);
  m.impl("binomial_cdf_backward", &binomial_cdf_backward);
  m.impl("binomial_pmf", &binomial_pmf);
  m.impl("binomial_pmf_backward", &binomial_pmf_backward);
}

}  // namespace torchscience::meta::probability
