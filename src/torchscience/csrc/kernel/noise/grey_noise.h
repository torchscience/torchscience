#pragma once

#include <ATen/ATen.h>
#include <ATen/Generator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

namespace torchscience::kernel::noise {

// Generate 1-D grey noise: white noise pre-emphasized by the inverse of the
// IEC 61672-1 A-weighting amplitude response. The resulting spectrum
// satisfies S(f) ~ 1 / R_A(f)^2, so when re-A-weighted (e.g. by the auditory
// system or a sound-level meter) the perceived spectrum is approximately
// flat.
//
// Algorithm:
//   1. Draw N i.i.d. samples from N(0, 1).
//   2. Take rfft.
//   3. Multiply each non-DC bin by 1 / R_A(f) (with a small floor on R_A to
//      avoid division by zero at f close to 0).
//   4. Set the DC bin to zero. R_A(0) = 0 (the formula has an f^4 factor in
//      the numerator), so 1/R_A is undefined at DC; zeroing also makes the
//      time-domain output exactly zero-mean.
//   5. irfft, then normalize by max absolute value.
//
// IEC 61672-1 A-weighting magnitude response (un-offset; the +2.00 dB
// normalization that makes A(1000 Hz) = 0 dB is a constant overall
// multiplier and cancels in every property used downstream):
//
//     R_A(f) =                 (12194^2) * f^4
//             ----------------------------------------------------
//             (f^2 + 20.6^2) * sqrt((f^2 + 107.7^2)(f^2 + 737.9^2))
//                          * (f^2 + 12194^2)
//
// Pole locations are stored as squared values (the formula only needs
// f^2 + p_i^2) for numerical efficiency.
inline void grey_noise(
    at::Tensor& out,
    c10::optional<at::Generator> generator,
    double sample_rate
) {
    TORCH_CHECK(out.dim() == 1, "grey_noise: internal tensor must be 1-D");
    TORCH_CHECK(out.is_cpu(), "grey_noise: CPU tensor expected");
    TORCH_CHECK(sample_rate > 0.0,
                "grey_noise: sample_rate must be positive, got ", sample_rate);
    const int64_t size = out.size(0);

    // IEC 61672-1 A-weighting pole frequencies (squared, in Hz^2).
    constexpr double kPole1Sq = 20.598997 * 20.598997;
    constexpr double kPole2Sq = 107.65265 * 107.65265;
    constexpr double kPole3Sq = 737.86223 * 737.86223;
    constexpr double kPole4Sq = 12194.217 * 12194.217;

    auto compute_float = [&](at::Tensor& dst) {
        at::Tensor white = at::empty_like(dst);
        white.normal_(0.0, 1.0, generator);

        at::Tensor spec = at::fft_rfft(white);
        // rfftfreq with d = 1/sample_rate gives bin centers in Hz.
        at::Tensor freqs = at::fft_rfftfreq(
            size, 1.0 / sample_rate, white.options()
        );

        at::Tensor f2 = freqs.pow(2);
        at::Tensor f4 = f2.pow(2);
        at::Tensor numerator = kPole4Sq * f4;
        at::Tensor denominator =
            (f2 + kPole1Sq)
            * ((f2 + kPole2Sq) * (f2 + kPole3Sq)).sqrt()
            * (f2 + kPole4Sq);
        at::Tensor r_a = numerator / denominator;

        // Inverse A-weighting; clamp R_A from below to avoid division by
        // zero at near-DC bins. The DC bin itself is then explicitly zeroed.
        at::Tensor inv_r_a = at::full({}, 1.0, r_a.options())
                             / r_a.clamp_min(1e-12);
        at::Tensor filtered = spec * inv_r_a;
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
