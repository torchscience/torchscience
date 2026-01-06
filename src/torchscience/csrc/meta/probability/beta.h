#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor beta_cdf(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto result_shape = at::infer_size(x.sizes(), a.sizes());
  result_shape = at::infer_size(result_shape, b.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> beta_cdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(a),
      at::empty_like(b));
}

at::Tensor beta_pdf(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto result_shape = at::infer_size(x.sizes(), a.sizes());
  result_shape = at::infer_size(result_shape, b.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> beta_pdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(a),
      at::empty_like(b));
}

at::Tensor beta_ppf(
    const at::Tensor& p,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto result_shape = at::infer_size(p.sizes(), a.sizes());
  result_shape = at::infer_size(result_shape, b.sizes());
  return at::empty(result_shape, p.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> beta_ppf_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& a,
    const at::Tensor& b) {
  return std::make_tuple(
      at::empty_like(p),
      at::empty_like(a),
      at::empty_like(b));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("beta_cdf", &beta_cdf);
  m.impl("beta_cdf_backward", &beta_cdf_backward);
  m.impl("beta_pdf", &beta_pdf);
  m.impl("beta_pdf_backward", &beta_pdf_backward);
  m.impl("beta_ppf", &beta_ppf);
  m.impl("beta_ppf_backward", &beta_ppf_backward);
}

}  // namespace torchscience::meta::probability
