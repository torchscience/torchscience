#pragma once

#include <tuple>

#include "chebyshev_polynomial_v.h"
#include "chebyshev_polynomial_w.h"

namespace torchscience::kernel::special_functions {

template <typename T>
C10_HOST_DEVICE std::tuple<T, T> chebyshev_polynomial_w_backward(T gradient, T x, T n) {
  T gradient_x;

  if (std::abs(x + T(1)) < T(1e-10)) {
    // Boundary at x = -1: use L'Hopital's rule
    // dW_n/dx at x = -1 = (-1)^n * (2n+1) * n * (n+1) / 3
    T sign;

    if (static_cast<int>(n) % 2 == 0) {
      sign = T(1);
    } else {
      sign = T(-1);
    }

    gradient_x = gradient * sign * (T(2) * n + T(1)) * n * (n + T(1)) / T(3);
  } else {
    // General case: dW_n/dx = ((2n+1) * V_n(x) - W_n(x)) / (2 * (x + 1))
    gradient_x = gradient * ((T(2) * n + T(1)) * chebyshev_polynomial_v(x, n) - chebyshev_polynomial_w(x, n)) / (T(2) * (x + T(1)));
  }

  return {
    gradient_x,
    T(0)
  };
}

} // namespace torchscience::kernel::special_functions
