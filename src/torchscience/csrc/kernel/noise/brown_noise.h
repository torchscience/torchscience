#pragma once

#include <ATen/ATen.h>
#include <ATen/Generator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

namespace torchscience::kernel::noise {

// Generate 1-D brown(ian) noise: PSD ~ 1/f^2 (alpha = -2).
//
// Spectral analogue of a bandlimited Wiener / random-walk process. Compared
// to pink noise the high frequencies are damped twice as strongly.
//
// Algorithm:
//   1. Draw N i.i.d. samples from N(0, 1).
//   2. Take the real FFT.
//   3. Divide each non-DC bin by |f|, giving |X(f)| ~ 1/f and hence |X(f)|^2
//      ~ 1/f^2.
//   4. Set the DC bin (f=0) to zero. The 1/f^2 law is undefined at f=0;
//      zeroing DC is the standard treatment and makes the time-domain output
//      exactly zero-mean.
//   5. Inverse FFT.
//   6. Normalize by max absolute value so peak amplitude is 1.
inline void brown_noise(
    at::Tensor& out,
    c10::optional<at::Generator> generator
) {
    TORCH_CHECK(out.dim() == 1, "brown_noise: internal tensor must be 1-D");
    TORCH_CHECK(out.is_cpu(), "brown_noise: CPU tensor expected");
    const int64_t size = out.size(0);

    auto compute_float = [&](at::Tensor& dst) {
        at::Tensor white = at::empty_like(dst);
        white.normal_(0.0, 1.0, generator);

        at::Tensor spec = at::fft_rfft(white);
        at::Tensor freqs = at::fft_rfftfreq(size, 1.0, white.options());
        at::Tensor scales = at::maximum(
            freqs.abs(),
            at::full({}, 1e-6, freqs.options())
        );
        at::Tensor filtered = spec / scales;
        // 1/f^2 at f=0 is undefined; zero DC explicitly.
        filtered.index_put_({0}, at::zeros({}, filtered.options()));
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
