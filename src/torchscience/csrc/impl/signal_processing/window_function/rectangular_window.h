#pragma once

#include <c10/macros/Macros.h>

namespace torchscience::impl::window_function {

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE void rectangular_window_kernel(
  scalar_t* output,
  int64_t numel,
  int64_t n
) {
  (void)n;  // n == numel for 1D output
  for (int64_t i = 0; i < numel; ++i) {
    output[i] = scalar_t(1);
  }
}

}  // namespace torchscience::impl::window_function
