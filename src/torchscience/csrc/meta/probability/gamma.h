#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor gamma_cdf(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto result_shape = at::infer_size(x.sizes(), shape.sizes());
  result_shape = at::infer_size(result_shape, scale.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_cdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(shape),
      at::empty_like(scale));
}

at::Tensor gamma_pdf(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto result_shape = at::infer_size(x.sizes(), shape.sizes());
  result_shape = at::infer_size(result_shape, scale.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_pdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(shape),
      at::empty_like(scale));
}

at::Tensor gamma_ppf(
    const at::Tensor& p,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto result_shape = at::infer_size(p.sizes(), shape.sizes());
  result_shape = at::infer_size(result_shape, scale.sizes());
  return at::empty(result_shape, p.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_ppf_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(p),
      at::empty_like(shape),
      at::empty_like(scale));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("gamma_cdf", &gamma_cdf);
  m.impl("gamma_cdf_backward", &gamma_cdf_backward);
  m.impl("gamma_pdf", &gamma_pdf);
  m.impl("gamma_pdf_backward", &gamma_pdf_backward);
  m.impl("gamma_ppf", &gamma_ppf);
  m.impl("gamma_ppf_backward", &gamma_ppf_backward);
}

}  // namespace torchscience::meta::probability
