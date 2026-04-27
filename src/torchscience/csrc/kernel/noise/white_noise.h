#pragma once

#include <ATen/ATen.h>
#include <ATen/Generator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

namespace torchscience::kernel::noise {

// Generate 1-D white noise: PSD ~ const (alpha = 0).
//
// Algorithm (FFT path, kept identical in structure to the other colors so that
// every "color" of noise satisfies the same zero-mean and unit-max-abs
// contracts):
//   1. Draw N i.i.d. samples from N(0, 1).
//   2. Take the real FFT.
//   3. Zero the DC bin so the time-domain output is exactly zero-mean.
//   4. Inverse FFT.
//   5. Normalize by max absolute value so peak amplitude is 1.
//
// Steps 2-4 are equivalent to subtracting the empirical mean. They preserve
// the white spectrum on every non-DC bin.
inline void white_noise(
    at::Tensor& out,
    c10::optional<at::Generator> generator
) {
    TORCH_CHECK(out.dim() == 1, "white_noise: internal tensor must be 1-D");
    TORCH_CHECK(out.is_cpu(), "white_noise: CPU tensor expected");
    const int64_t size = out.size(0);

    auto compute_float = [&](at::Tensor& dst) {
        at::Tensor white = at::empty_like(dst);
        white.normal_(0.0, 1.0, generator);

        at::Tensor spec = at::fft_rfft(white);
        // Zero DC -> exact zero-mean time series.
        spec.index_put_({0}, at::zeros({}, spec.options()));
        at::Tensor noise = at::fft_irfft(spec, size, -1, c10::nullopt);
        at::Tensor mx = noise.abs().max();
        noise = noise / mx.clamp_min(1e-12);
        dst.copy_(noise);
    };

    if (out.scalar_type() == at::kHalf || out.scalar_type() == at::kBFloat16) {
        at::Tensor tmp = at::empty({size}, out.options().dtype(at::kFloat));
        compute_float(tmp);
        out.copy_(tmp);
    } else {
        compute_float(out);
    }
}

}  // namespace torchscience::kernel::noise
