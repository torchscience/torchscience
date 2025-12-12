#pragma once

#include <boost/math/special_functions/cos_pi.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <c10/util/MathConstants.h>
#include <c10/util/complex.h>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T sinc_pi(T x) {
  return boost::math::sinc_pi(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> sinc_pi(c10::complex<T> z) {
  return boost::math::sinc_pi(z);
}

template <typename T>
C10_HOST_DEVICE T sinc_pi_backward(T x) {
  // d/dx sinc_pi(x) = d/dx [sin(pi*x)/(pi*x)]
  // = (pi*cos(pi*x) - sin(pi*x)/x) / (pi*x)
  // = (cos(pi*x) - sinc_pi(x)) * pi / (pi*x)
  // = (cos_pi(x) - sinc_pi(x)) / x
  // At x=0, the derivative is 0 (maximum of sinc_pi)
  if (x == T{0}) {
    return T{0};
  }
  return (boost::math::cos_pi(x) - boost::math::sinc_pi(x)) / x;
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> sinc_pi_backward(c10::complex<T> z) {
  // Same formula for complex
  if (z == c10::complex<T>{0}) {
    return c10::complex<T>{0};
  }
  return (boost::math::cos_pi(z) - boost::math::sinc_pi(z)) / z;
}

} // namespace torchscience::impl::special_functions
