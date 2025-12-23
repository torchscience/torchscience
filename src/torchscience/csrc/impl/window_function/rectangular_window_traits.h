#pragma once

#include <vector>

namespace torchscience::impl::window_function {

struct RectangularWindowTraits {
    static std::vector<int64_t> output_shape(int64_t n) {
        return {n};
    }

    template<typename scalar_t>
    static void kernel(scalar_t* output, int64_t numel, int64_t n) {
        (void)n;  // n == numel for rectangular window
        for (int64_t i = 0; i < numel; ++i) {
            output[i] = scalar_t(1);
        }
    }
};

}  // namespace torchscience::impl::window_function
