#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "cmath_compat.h"
#include "sin_pi.h"
#include "cos_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
C10_HOST_DEVICE T tetragamma(T z) {
  if (cmath_compat::isnan(z)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (cmath_compat::isinf(z)) {
    return z > T(0) ? T(0) : std::numeric_limits<T>::quiet_NaN();
  }

  // Reflection formula for negative args:
  // psi''(z) = 2*pi^3*cos(pi*z)/sin^3(pi*z) - psi''(1-z)
  if (z < T(0)) {
    if (z == std::floor(z)) {
      return std::numeric_limits<T>::quiet_NaN();
    }

    T s = sin_pi(z);
    T c = cos_pi(z);
    return tetragamma(T(1) - z) - static_cast<T>(2.0 * M_PI * M_PI * M_PI) * c / (s * s * s);
  }

  T psi2 = T(0);

  T y = z;

  while (y < T(6)) {
    psi2 -= T(2) / (y * y * y);

    y += T(1);
  }

  T y2 = T(1) / (y * y);
  T y3 = y2 / y;

  // Asymptotic expansion: psi''(y) = -1/y^2 - 1/y^3 - 1/(2y^4) + 1/(6y^6) - 1/(6y^8) + 3/(10y^10) - 5/(6y^12) + ...
  psi2 += -y2 - y3 - y2 * y2 * (T(0.5) - y2 * (T(1.0/6) - y2 * (T(1.0/6) - y2 * (T(3.0/10) - y2 * T(5.0/6)))));

  return psi2;
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> tetragamma(c10::complex<T> z) {
  if (!cmath_compat::isfinite(z.real()) || !cmath_compat::isfinite(z.imag())) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN());
  }

  c10::complex<T> psi2(T(0), T(0));

  c10::complex<T> y = z;

  if (y.real() < T(0.5)) {
    return tetragamma(c10::complex<T>(T(1), T(0)) - y) - static_cast<T>(M_PI * M_PI * M_PI) * static_cast<T>(2) * cos_pi(y) / (sin_pi(y) * sin_pi(y) * sin_pi(y));
  }

  while (std::abs(y) < T(6)) {
    psi2 = psi2 - c10::complex<T>(T(2), T(0)) / (y * y * y);

    y = y + c10::complex<T>(T(1), T(0));
  }

  c10::complex<T> y2 = c10::complex<T>(T(1), T(0)) / (y * y);
  c10::complex<T> y3 = y2 / y;

  return psi2 + (-y2 - y3 - y2 * y2 * (c10::complex<T>(T(0.5), T(0)) - y2 * (c10::complex<T>(T(1.0/6), T(0)) - y2 * (c10::complex<T>(T(1.0/6), T(0)) - y2 * (c10::complex<T>(T(3.0/10), T(0)) - y2 * c10::complex<T>(T(5.0/6), T(0)))))));
}

} // namespace torchscience::kernel::special_functions
