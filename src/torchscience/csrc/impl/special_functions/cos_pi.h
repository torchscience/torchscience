#pragma once

#include <boost/math/special_functions/cos_pi.hpp>
#include <boost/math/special_functions/sin_pi.hpp>
#include <c10/util/MathConstants.h>
#include <c10/util/complex.h>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T cos_pi(T x) {
  return boost::math::cos_pi(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> cos_pi(c10::complex<T> z) {
  return boost::math::cos_pi(z);
}

template <typename T>
C10_HOST_DEVICE T cos_pi_backward(T x) {
  return -c10::pi<T> * boost::math::sin_pi(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> cos_pi_backward(c10::complex<T> z) {
  return -c10::pi<T> * boost::math::sin_pi(z);
}

} // namespace torchscience::impl::special_functions
