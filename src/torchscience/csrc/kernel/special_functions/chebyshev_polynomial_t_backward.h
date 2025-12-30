#pragma once

#include <tuple>

#include "chebyshev_polynomial_u.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> chebyshev_polynomial_t_backward(T gradient, T x, T n) {
  T gradient_x;

  if (n > T(0)) {
    gradient_x = gradient * n * chebyshev_polynomial_u(x, n - T(1));
  } else {
    gradient_x = T(0);
  }

  return {
    gradient_x,
    T(0)
  };
}

} // namespace torchscience::kernel::special_functions
