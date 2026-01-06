#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor f_cdf(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto result_shape = at::infer_size(x.sizes(), dfn.sizes());
  result_shape = at::infer_size(result_shape, dfd.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_cdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(dfn),
      at::empty_like(dfd));
}

at::Tensor f_pdf(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto result_shape = at::infer_size(x.sizes(), dfn.sizes());
  result_shape = at::infer_size(result_shape, dfd.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_pdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(dfn),
      at::empty_like(dfd));
}

at::Tensor f_ppf(
    const at::Tensor& p,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto result_shape = at::infer_size(p.sizes(), dfn.sizes());
  result_shape = at::infer_size(result_shape, dfd.sizes());
  return at::empty(result_shape, p.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_ppf_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  return std::make_tuple(
      at::empty_like(p),
      at::empty_like(dfn),
      at::empty_like(dfd));
}

at::Tensor f_sf(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto result_shape = at::infer_size(x.sizes(), dfn.sizes());
  result_shape = at::infer_size(result_shape, dfd.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_sf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(dfn),
      at::empty_like(dfd));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("f_cdf", &f_cdf);
  m.impl("f_cdf_backward", &f_cdf_backward);
  m.impl("f_pdf", &f_pdf);
  m.impl("f_pdf_backward", &f_pdf_backward);
  m.impl("f_ppf", &f_ppf);
  m.impl("f_ppf_backward", &f_ppf_backward);
  m.impl("f_sf", &f_sf);
  m.impl("f_sf_backward", &f_sf_backward);
}

}  // namespace torchscience::meta::probability
