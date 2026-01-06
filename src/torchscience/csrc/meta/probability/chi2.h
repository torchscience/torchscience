#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor chi2_cdf(
    const at::Tensor& x,
    const at::Tensor& df) {
  // Infer broadcast shape
  auto output_shape = at::infer_size(x.sizes(), df.sizes());
  auto output_dtype = at::result_type(x, df);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor> chi2_cdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& df) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(df));
}

at::Tensor chi2_pdf(
    const at::Tensor& x,
    const at::Tensor& df) {
  auto output_shape = at::infer_size(x.sizes(), df.sizes());
  auto output_dtype = at::result_type(x, df);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor> chi2_pdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& df) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(df));
}

at::Tensor chi2_ppf(
    const at::Tensor& p,
    const at::Tensor& df) {
  auto output_shape = at::infer_size(p.sizes(), df.sizes());
  auto output_dtype = at::result_type(p, df);
  return at::empty(output_shape, p.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor> chi2_ppf_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& df) {
  return std::make_tuple(
      at::empty_like(p),
      at::empty_like(df));
}

at::Tensor chi2_sf(
    const at::Tensor& x,
    const at::Tensor& df) {
  auto output_shape = at::infer_size(x.sizes(), df.sizes());
  auto output_dtype = at::result_type(x, df);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor> chi2_sf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& df) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(df));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("chi2_cdf", &chi2_cdf);
  m.impl("chi2_cdf_backward", &chi2_cdf_backward);
  m.impl("chi2_pdf", &chi2_pdf);
  m.impl("chi2_pdf_backward", &chi2_pdf_backward);
  m.impl("chi2_ppf", &chi2_ppf);
  m.impl("chi2_ppf_backward", &chi2_ppf_backward);
  m.impl("chi2_sf", &chi2_sf);
  m.impl("chi2_sf_backward", &chi2_sf_backward);
}

}  // namespace torchscience::meta::probability
