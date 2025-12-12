#pragma once

#include <c10/util/MathConstants.h>
#include <c10/util/complex.h>

namespace torchscience::impl::special_functions {

// Forward declarations
template <typename T>
C10_HOST_DEVICE T cos_pi(T x);

template <typename T>
C10_HOST_DEVICE c10::complex<T> cos_pi(c10::complex<T> z);

template <typename T>
C10_HOST_DEVICE T sin_pi(T x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (x < T{0}) {
    return -sin_pi(-x);
  }

  if (x < T{0.5}) {
    return std::sin(x * c10::pi<T>);
  }

  if (x < T{1}) {
    return std::sin((T{1} - x) * c10::pi<T>);
  }

  auto n = std::floor(x);

  auto arg = x - n;

  auto sign = (static_cast<int>(n) & 1) == 1 ? T{-1} : T{1};

  auto sinval = arg < T{0.5} ? sin_pi(arg) : sin_pi(T{1} - arg);

  return sign * sinval;
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> sin_pi(c10::complex<T> z) {
  return sin_pi(std::real(z)) * std::cosh(c10::pi<T> * std::imag(z)) + c10::complex<T>{0, 1} * cos_pi(std::real(z)) * std::sinh(c10::pi<T> * std::imag(z));
}

template <typename T>
C10_HOST_DEVICE T sin_pi_backward(T x) {
  return c10::pi<T> * cos_pi(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> sin_pi_backward(c10::complex<T> z) {
  return c10::pi<T> * cos_pi(z);
}

}

// Include cos_pi definitions (after all forward declarations are satisfied)
#include <torchscience/csrc/impl/special_functions/cos_pi.h>
