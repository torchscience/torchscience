#pragma once

#include <c10/macros/Macros.h>
#include <cmath>

namespace torchscience::impl::waveform {

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE void sine_wave_kernel(
  scalar_t* output,
  int64_t numel,
  int64_t n,
  double frequency,
  double sample_rate,
  double amplitude,
  double phase
) {
  (void)n;  // n == numel for 1D output
  const double angular_freq = 2.0 * M_PI * frequency / sample_rate;

  for (int64_t i = 0; i < numel; ++i) {
    output[i] = static_cast<scalar_t>(
      amplitude * std::sin(angular_freq * static_cast<double>(i) + phase)
    );
  }
}

}  // namespace torchscience::impl::waveform
