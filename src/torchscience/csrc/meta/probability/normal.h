#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor normal_cdf(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Infer broadcast shape
  auto shape1 = at::infer_size(x.sizes(), loc.sizes());
  auto output_shape = at::infer_size(shape1, scale.sizes());
  auto output_dtype = at::result_type(x, loc);
  output_dtype = at::result_type(at::empty({}, x.options().dtype(output_dtype)), scale);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_cdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(loc),
      at::empty_like(scale));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> normal_cdf_backward_backward(
    const at::Tensor& gg_x,
    const at::Tensor& gg_loc,
    const at::Tensor& gg_scale,
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(grad),
      at::empty_like(x),
      at::empty_like(loc),
      at::empty_like(scale));
}

at::Tensor normal_pdf(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  auto shape1 = at::infer_size(x.sizes(), loc.sizes());
  auto output_shape = at::infer_size(shape1, scale.sizes());
  auto output_dtype = at::result_type(x, loc);
  output_dtype = at::result_type(at::empty({}, x.options().dtype(output_dtype)), scale);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_pdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(loc),
      at::empty_like(scale));
}

at::Tensor normal_ppf(
    const at::Tensor& p,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  auto shape1 = at::infer_size(p.sizes(), loc.sizes());
  auto output_shape = at::infer_size(shape1, scale.sizes());
  auto output_dtype = at::result_type(p, loc);
  output_dtype = at::result_type(at::empty({}, p.options().dtype(output_dtype)), scale);
  return at::empty(output_shape, p.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_ppf_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(p),
      at::empty_like(loc),
      at::empty_like(scale));
}

at::Tensor normal_sf(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  auto shape1 = at::infer_size(x.sizes(), loc.sizes());
  auto output_shape = at::infer_size(shape1, scale.sizes());
  auto output_dtype = at::result_type(x, loc);
  output_dtype = at::result_type(at::empty({}, x.options().dtype(output_dtype)), scale);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_sf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(loc),
      at::empty_like(scale));
}

at::Tensor normal_logpdf(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  auto shape1 = at::infer_size(x.sizes(), loc.sizes());
  auto output_shape = at::infer_size(shape1, scale.sizes());
  auto output_dtype = at::result_type(x, loc);
  output_dtype = at::result_type(at::empty({}, x.options().dtype(output_dtype)), scale);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_logpdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(loc),
      at::empty_like(scale));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("normal_cdf", &normal_cdf);
  m.impl("normal_cdf_backward", &normal_cdf_backward);
  m.impl("normal_cdf_backward_backward", &normal_cdf_backward_backward);
  m.impl("normal_pdf", &normal_pdf);
  m.impl("normal_pdf_backward", &normal_pdf_backward);
  m.impl("normal_ppf", &normal_ppf);
  m.impl("normal_ppf_backward", &normal_ppf_backward);
  m.impl("normal_sf", &normal_sf);
  m.impl("normal_sf_backward", &normal_sf_backward);
  m.impl("normal_logpdf", &normal_logpdf);
  m.impl("normal_logpdf_backward", &normal_logpdf_backward);
}

}  // namespace torchscience::meta::probability
