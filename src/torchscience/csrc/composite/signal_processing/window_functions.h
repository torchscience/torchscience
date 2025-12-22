#pragma once

#include <torch/extension.h>
#include "../../impl/signal_processing/window_function/rectangular_window.h"

namespace torchscience::window_function {

inline at::Tensor rectangular_window(
  int64_t n,
  const c10::optional<at::ScalarType> dtype,
  const c10::optional<at::Layout> layout,
  const c10::optional<at::Device> device,
  const bool requires_grad
) {
  TORCH_CHECK(n >= 0, "rectangular_window: n must be non-negative, got ", n);

  auto options = at::TensorOptions()
    .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::kCPU))
    .requires_grad(false);

  at::Tensor output = at::empty({n}, options);

  if (n > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      output.scalar_type(),
      "rectangular_window",
      [&]() {
        impl::window_function::rectangular_window_kernel<scalar_t>(
          output.data_ptr<scalar_t>(),
          n,
          n
        );
      }
    );
  }

  if (requires_grad) {
    output = output.requires_grad_(true);
  }

  return output;
}

} // namespace torchscience::window_function

TORCH_LIBRARY_IMPL(torchscience, CompositeExplicitAutograd, module) {
  module.impl(
    "rectangular_window",
    &torchscience::window_function::rectangular_window
  );
}
