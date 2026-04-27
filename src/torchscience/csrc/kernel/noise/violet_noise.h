#pragma once

#include <ATen/ATen.h>
#include <ATen/Generator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

namespace torchscience::kernel::noise {

// Generate 1-D violet (purple) noise: PSD ~ f^2 (alpha = +2).
//
// Spectral mirror of brown noise. Equivalently, violet noise is the
// discrete-time derivative of white noise.
//
// Algorithm:
//   1. Draw N i.i.d. samples from N(0, 1).
//   2. Take the real FFT.
//   3. Multiply each bin by |f|, giving |X(f)| ~ f and hence |X(f)|^2 ~ f^2.
//   4. The DC bin is naturally zero because the filter f vanishes at f=0;
//      this makes the time-domain output exactly zero-mean.
//   5. Inverse FFT.
//   6. Normalize by max absolute value so peak amplitude is 1.
inline void violet_noise(
    at::Tensor& out,
    c10::optional<at::Generator> generator
) {
    TORCH_CHECK(out.dim() == 1, "violet_noise: internal tensor must be 1-D");
    TORCH_CHECK(out.is_cpu(), "violet_noise: CPU tensor expected");
    const int64_t size = out.size(0);

    auto compute_float = [&](at::Tensor& dst) {
        at::Tensor white = at::empty_like(dst);
        white.normal_(0.0, 1.0, generator);

        at::Tensor spec = at::fft_rfft(white);
        at::Tensor freqs = at::fft_rfftfreq(size, 1.0, white.options());
        at::Tensor filter_amp = freqs.abs();
        at::Tensor filtered = spec * filter_amp;
        at::Tensor noise = at::fft_irfft(filtered, size, -1, c10::nullopt);
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
