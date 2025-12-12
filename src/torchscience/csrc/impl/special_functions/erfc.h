#pragma once

#include <boost/math/special_functions/erf.hpp>
#include <boost/math/constants/constants.hpp>
#include <c10/util/complex.h>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T erfc(T x) {
  return boost::math::erfc(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> erfc(c10::complex<T> z) {
  return boost::math::erfc(z);
}

template <typename T>
C10_HOST_DEVICE T erfc_backward(T x) {
  T two_over_sqrt_pi = T{2} / boost::math::constants::root_pi<T>();
  return -two_over_sqrt_pi * std::exp(-x * x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> erfc_backward(c10::complex<T> z) {
  T two_over_sqrt_pi = T{2} / boost::math::constants::root_pi<T>();
  return -two_over_sqrt_pi * std::exp(-z * z);
}

} // namespace torchscience::impl::special_functions
