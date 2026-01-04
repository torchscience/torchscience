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

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("normal_cdf", &normal_cdf);
}

}  // namespace torchscience::meta::probability
