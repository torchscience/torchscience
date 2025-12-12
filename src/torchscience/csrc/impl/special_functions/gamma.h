#pragma once

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <c10/util/complex.h>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T gamma(T x) {
  return boost::math::tgamma(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> gamma(c10::complex<T> z) {
  return boost::math::tgamma(z);
}

template <typename T>
C10_HOST_DEVICE T gamma_backward(T x) {
  return boost::math::tgamma(x) * boost::math::digamma(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> gamma_backward(c10::complex<T> z) {
  return boost::math::tgamma(z) * boost::math::digamma(z);
}

} // namespace torchscience::impl::special_functions
