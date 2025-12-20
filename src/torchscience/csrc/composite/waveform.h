#pragma once

#include <torch/extension.h>

#include <cmath>

namespace torchscience::waveform {

inline at::Tensor sine_wave(
  int64_t n,
  double frequency,
  double sample_rate,
  double amplitude,
  double phase,
  const c10::optional<at::ScalarType> dtype,
  const c10::optional<at::Layout> layout,
  const c10::optional<at::Device> device,
  const bool requires_grad
) {
  TORCH_CHECK(n >= 0, "sine_wave: n must be non-negative, got ", n);
  TORCH_CHECK(sample_rate > 0, "sine_wave: sample_rate must be positive, got ", sample_rate);

  if (n == 0) {
    return at::empty(
      {0},
      at::TensorOptions()
        .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
        .layout(layout.value_or(at::kStrided))
        .device(device.value_or(at::kCPU))
        .requires_grad(requires_grad)
    );
  }

  auto options = at::TensorOptions()
    .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::kCPU))
    .requires_grad(false);

  // Create time indices: [0, 1, 2, ..., n-1]
  at::Tensor t = at::arange(n, options);

  // Compute angular frequency: 2 * pi * frequency / sample_rate
  double angular_freq = 2.0 * M_PI * frequency / sample_rate;

  // Compute sine wave: amplitude * sin(angular_freq * t + phase)
  at::Tensor result = amplitude * at::sin(angular_freq * t + phase);

  if (requires_grad) {
    result = result.requires_grad_(true);
  }

  return result;
}

} // namespace torchscience::waveform

TORCH_LIBRARY_IMPL(torchscience, CompositeImplicitAutograd, module) {
  module.impl(
    "sine_wave",
    &torchscience::waveform::sine_wave
  );
}
