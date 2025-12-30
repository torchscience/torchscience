#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

template <typename T>
T tanh_pi(T x) {
  return std::tanh(static_cast<T>(M_PI) * x);
}

template <typename T>
c10::complex<T> tanh_pi(c10::complex<T> z) {
  T x = z.real();
  T y = z.imag();

  T two_pi_x = static_cast<T>(2 * M_PI) * x;
  T two_pi_y = static_cast<T>(2 * M_PI) * y;

  T sinh_2x = std::sinh(two_pi_x);
  T cosh_2x = std::cosh(two_pi_x);

  T sin_2y = std::sin(two_pi_y);
  T cos_2y = std::cos(two_pi_y);

  T denom = cosh_2x + cos_2y;

  return c10::complex<T>(sinh_2x / denom, sin_2y / denom);
}

} // namespace torchscience::kernel::special_functions
