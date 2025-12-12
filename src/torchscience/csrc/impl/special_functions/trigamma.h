#pragma once

#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <c10/util/complex.h>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T trigamma(T x) {
  return boost::math::trigamma(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> trigamma(c10::complex<T> z) {
  return boost::math::trigamma(z);
}

template <typename T>
C10_HOST_DEVICE T trigamma_backward(T x) {
  return boost::math::polygamma(2, x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> trigamma_backward(c10::complex<T> z) {
  return boost::math::polygamma(2, z);
}

} // namespace torchscience::impl::special_functions
