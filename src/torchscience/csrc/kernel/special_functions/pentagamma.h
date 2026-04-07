#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "sin_pi.h"
#include "cos_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T pentagamma(T z) {
  if (std::isnan(z)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (std::isinf(z)) {
    return z > T(0) ? T(0) : std::numeric_limits<T>::quiet_NaN();
  }

  // Reflection formula for negative args:
  // psi'''(z) = -2*pi^4*(1 + 2*cos^2(pi*z))/sin^4(pi*z) + psi'''(1-z)
  if (z < T(0)) {
    if (z == std::floor(z)) {
      return std::numeric_limits<T>::quiet_NaN();
    }

    T s = sin_pi(z);
    T c = cos_pi(z);
    T s2 = s * s;
    T s4 = s2 * s2;
    T c2 = c * c;
    T pi4 = static_cast<T>(M_PI * M_PI * M_PI * M_PI);

    return T(2) * pi4 * (T(1) + T(2) * c2) / s4 - pentagamma(T(1) - z);
  }

  T psi3 = T(0);

  T y = z;

  while (y < T(6)) {
    psi3 += T(6) / (y * y * y * y);

    y += T(1);
  }

  T y2 = T(1) / (y * y);
  T y3 = y2 / y;
  T y4 = y2 * y2;

  // Asymptotic expansion: psi'''(y) = 2/y^3 + 3/y^4 + 2/y^5 - 1/y^7 + 4/(3y^9) - 3/y^11 + 10/y^13 - ...
  return psi3 + (T(2) * y3 + T(3) * y4 + T(2) * y4 / y + y3 * y4 * (T(-1) + y2 * (T(4.0/3) + y2 * (T(-3) + y2 * T(10)))));
}

template <typename T>
c10::complex<T> pentagamma(c10::complex<T> z) {
  if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN());
  }

  c10::complex<T> psi3(T(0), T(0));

  c10::complex<T> y = z;

  if (y.real() < T(0.5)) {
    // Reflection formula: psi'''(z) - psi'''(1-z) = -2*pi^4*(1 + 2*cos^2(pi*z))/sin^4(pi*z)
    auto sin_piz = sin_pi(y);
    auto cos_piz = cos_pi(y);
    auto sin_piz_sq = sin_piz * sin_piz;
    auto sin_piz_fourth = sin_piz_sq * sin_piz_sq;
    auto cos_piz_sq = cos_piz * cos_piz;
    auto pi_fourth = static_cast<T>(M_PI * M_PI * M_PI * M_PI);

    return pi_fourth * static_cast<T>(2) * (c10::complex<T>(T(1), T(0)) + static_cast<T>(2) * cos_piz_sq) / sin_piz_fourth - pentagamma(c10::complex<T>(T(1), T(0)) - y);
  }

  while (std::abs(y) < T(6)) {
    psi3 = psi3 + c10::complex<T>(T(6), T(0)) / (y * y * y * y);

    y = y + c10::complex<T>(T(1), T(0));
  }

  c10::complex<T> y2 = c10::complex<T>(T(1), T(0)) / (y * y);
  c10::complex<T> y3 = y2 / y;
  c10::complex<T> y4 = y2 * y2;

  return psi3 + (c10::complex<T>(T(2), T(0)) * y3 + c10::complex<T>(T(3), T(0)) * y4 + c10::complex<T>(T(2), T(0)) * y4 / y + y3 * y4 * (c10::complex<T>(T(-1), T(0)) + y2 * (c10::complex<T>(T(4.0/3), T(0)) + y2 * (c10::complex<T>(T(-3), T(0)) + y2 * c10::complex<T>(T(10), T(0))))));
}

} // namespace torchscience::kernel::special_functions
