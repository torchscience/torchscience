#pragma once

#include <cmath>
#include <vector>

namespace torchscience::impl::waveform {

struct SineWaveTraits {
    static std::vector<int64_t> output_shape(
        int64_t n,
        double frequency,
        double sample_rate,
        double amplitude,
        double phase
    ) {
        (void)frequency;
        (void)sample_rate;
        (void)amplitude;
        (void)phase;
        return {n};
    }

    template<typename scalar_t>
    static void kernel(
        scalar_t* output,
        int64_t numel,
        int64_t n,
        double frequency,
        double sample_rate,
        double amplitude,
        double phase
    ) {
        (void)n;  // n == numel
        double angular_freq = 2.0 * M_PI * frequency / sample_rate;
        for (int64_t i = 0; i < numel; ++i) {
            output[i] = static_cast<scalar_t>(
                amplitude * std::sin(angular_freq * static_cast<double>(i) + phase)
            );
        }
    }
};

}  // namespace torchscience::impl::waveform
