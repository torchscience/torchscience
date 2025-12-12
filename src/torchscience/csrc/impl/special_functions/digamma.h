#pragma once

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <c10/util/complex.h>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T digamma(T x) {
  return boost::math::digamma(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> digamma(c10::complex<T> z) {
  return boost::math::digamma(z);
}

template <typename T>
C10_HOST_DEVICE T digamma_backward(T x) {
  return boost::math::trigamma(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> digamma_backward(c10::complex<T> z) {
  return boost::math::trigamma(z);
}

} // namespace torchscience::impl::special_functions
