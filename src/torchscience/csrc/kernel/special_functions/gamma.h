#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "sin_pi.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
constexpr T pole_tolerance();

template <>
constexpr float pole_tolerance<float>() { return 1e-6f; }

template <>
constexpr double pole_tolerance<double>() { return 1e-12; }

} // namespace detail

template <typename T>
T gamma(T z) {
  constexpr double kGammaG = 7.0;

  constexpr double coefficients[] = {
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  };

  if (z < T(0.5)) {
    return static_cast<T>(M_PI) / (std::sin(static_cast<T>(M_PI) * z) * gamma(T(1) - z));
  }

  T z_adj = z - T(1);

  T x = static_cast<T>(coefficients[0]);

  for (int i = 1; i < 9; ++i) {
    x += static_cast<T>(coefficients[i]) / (z_adj + T(i));
  }

  T t = z_adj + static_cast<T>(kGammaG) + T(0.5);

  return std::sqrt(static_cast<T>(2 * M_PI)) * std::pow(t, z_adj + T(0.5)) * std::exp(-t) * x;
}

template <typename T>
c10::complex<T> gamma(c10::complex<T> z) {
  constexpr double kGammaG = 7.0;

  constexpr double coefficients[] = {
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  };

  const T tol = detail::pole_tolerance<T>();

  if (std::abs(z.imag()) < tol) {
    T real_part = z.real();

    if (real_part <= T(0) && std::abs(real_part - std::round(real_part)) < tol) {
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }
  }

  if (z.real() < T(0.5)) {
    return static_cast<T>(M_PI) / (sin_pi(z) * gamma(c10::complex<T>(T(1), T(0)) - z));
  }

  c10::complex<T> z_adj = z - c10::complex<T>(T(1), T(0));

  c10::complex<T> x(static_cast<T>(coefficients[0]), T(0));

  for (int i = 1; i < 9; ++i) {
    x += static_cast<T>(coefficients[i]) / (z_adj + c10::complex<T>(T(i), T(0)));
  }

  c10::complex<T> t = z_adj + c10::complex<T>(static_cast<T>(kGammaG) + T(0.5), T(0));

  return std::sqrt(static_cast<T>(2 * M_PI)) * std::pow(t, z_adj + c10::complex<T>(T(0.5), T(0))) * std::exp(-t) * x;
}

} // namespace torchscience::kernel::special_functions
