#pragma once

#include <cmath>

namespace torchscience::kernel::special_functions {

template <typename T>
T chebyshev_polynomial_t(T x, T n) {
  if (std::abs(x) <= T(1)) {
    return std::cos(n * std::acos(x));
  }

  if (x > T(1)) {
    return std::cosh(n * std::acosh(x));
  }

  T sign;

  if (static_cast<int>(n) % 2 == 0) {
    sign = T(1);
  } else {
    sign = T(-1);
  }

  return sign * std::cosh(n * std::acosh(-x));
}

} // namespace torchscience::kernel::special_functions
