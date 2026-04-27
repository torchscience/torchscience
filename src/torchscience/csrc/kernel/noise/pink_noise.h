#pragma once

#include <ATen/ATen.h>
#include <ATen/Generator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

namespace torchscience::kernel::noise {

// Generate 1-D approximate pink noise via FFT filtering of Gaussian white noise.
// Matches the reference used in tests: rfft, scale by 1/max(sqrt(|f|), eps), irfft,
// then normalize by max absolute value.
inline void pink_noise(
    at::Tensor& out,
    c10::optional<at::Generator> generator
) {
    TORCH_CHECK(out.dim() == 1, "pink_noise: internal tensor must be 1-D");
    TORCH_CHECK(out.is_cpu(), "pink_noise: CPU tensor expected");
    const int64_t size = out.size(0);

    auto compute_float = [&](at::Tensor& dst) {
        at::Tensor white = at::empty_like(dst);
        white.normal_(0.0, 1.0, generator);

        at::Tensor spec = at::fft_rfft(white);
        at::Tensor freqs = at::fft_rfftfreq(size, 1.0, white.options());
        at::Tensor scales = at::maximum(
            freqs.abs().sqrt(),
            at::full({}, 1e-6, freqs.options())
        );
        at::Tensor filtered = spec / scales;
        at::Tensor pink = at::fft_irfft(filtered, size, -1, c10::nullopt);
        at::Tensor mx = pink.abs().max();
        pink = pink / mx.clamp_min(1e-12);
        dst.copy_(pink);
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
