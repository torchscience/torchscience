#pragma once

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <c10/util/complex.h>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T log_gamma(T x) {
  return boost::math::lgamma(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> log_gamma(c10::complex<T> z) {
  return boost::math::lgamma(z);
}

template <typename T>
C10_HOST_DEVICE T log_gamma_backward(T x) {
  return boost::math::digamma(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> log_gamma_backward(c10::complex<T> z) {
  return boost::math::digamma(z);
}

} // namespace torchscience::impl::special_functions
