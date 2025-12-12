#pragma once

#include <boost/math/special_functions/sinhc.hpp>
#include <c10/util/MathConstants.h>
#include <c10/util/complex.h>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T sinhc_pi(T x) {
  return boost::math::sinhc_pi(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> sinhc_pi(c10::complex<T> z) {
  return boost::math::sinhc_pi(z);
}

template <typename T>
C10_HOST_DEVICE T sinhc_pi_backward(T x) {
  // d/dx sinhc_pi(x) = d/dx [sinh(pi*x)/(pi*x)]
  // = (cosh(pi*x) - sinhc_pi(x)) / x
  // At x=0, the derivative is 0
  if (x == T{0}) {
    return T{0};
  }
  T pi_x = c10::pi<T> * x;
  return (std::cosh(pi_x) - boost::math::sinhc_pi(x)) / x;
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> sinhc_pi_backward(c10::complex<T> z) {
  // Same formula for complex
  if (z == c10::complex<T>{0}) {
    return c10::complex<T>{0};
  }
  c10::complex<T> pi_z = c10::pi<T> * z;
  return (std::cosh(pi_z) - boost::math::sinhc_pi(z)) / z;
}

} // namespace torchscience::impl::special_functions
