#pragma once

#include <torch/extension.h>

namespace torchscience::window_function {

inline at::Tensor rectangular_window(
  int64_t n,
  const c10::optional<at::ScalarType> dtype,
  const c10::optional<at::Layout> layout,
  const c10::optional<at::Device> device,
  const bool requires_grad
) {
  TORCH_CHECK(n >= 0, "rectangular_window: n must be non-negative, got ", n);

  return at::ones(
    {n},
    at::TensorOptions()
      .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
      .layout(layout.value_or(at::kStrided))
      .device(device.value_or(at::kCPU))
      .requires_grad(requires_grad)
  );
}

} // namespace torchscience::window_function

TORCH_LIBRARY_IMPL(torchscience, CompositeImplicitAutograd, module) {
  module.impl(
    "rectangular_window",
    &torchscience::window_function::rectangular_window
  );
}
