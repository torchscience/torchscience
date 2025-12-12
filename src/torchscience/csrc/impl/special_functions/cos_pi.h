#pragma once

#include <c10/util/MathConstants.h>
#include <c10/util/complex.h>

namespace torchscience::impl::special_functions {

// Forward declarations
template <typename T>
C10_HOST_DEVICE T sin_pi(T x);

template <typename T>
C10_HOST_DEVICE c10::complex<T> sin_pi(c10::complex<T> z);

template <typename T>
C10_HOST_DEVICE T cos_pi(T x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (x < T{0}) {
    return cos_pi(-x);
  }

  if (x < T{0.5}) {
    return std::cos(x * c10::pi<T>);
  }

  if (x < T{1}) {
    return -std::cos((T{1} - x) * c10::pi<T>);
  }

  auto n = std::floor(x);

  auto arg = x - n;

  auto sign = (static_cast<int>(n) & 1) == 1 ? T{-1} : T{1};

  return sign * cos_pi(arg);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> cos_pi(c10::complex<T> z) {
  return cos_pi(std::real(z)) * std::cosh(c10::pi<T> * std::imag(z)) - c10::complex<T>{0, 1} * sin_pi(std::real(z)) * std::sinh(c10::pi<T> * std::imag(z));
}

template <typename T>
C10_HOST_DEVICE T cos_pi_backward(T x) {
  return -c10::pi<T> * sin_pi(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> cos_pi_backward(c10::complex<T> z) {
  return -c10::pi<T> * sin_pi(z);
}

}
