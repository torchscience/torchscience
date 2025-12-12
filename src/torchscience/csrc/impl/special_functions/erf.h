#pragma once

#include <boost/math/special_functions/erf.hpp>
#include <c10/util/MathConstants.h>
#include <c10/util/complex.h>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T erf(T x) {
  return boost::math::erf(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> erf(c10::complex<T> z) {
  return boost::math::erf(z);
}

template <typename T>
C10_HOST_DEVICE T erf_backward(T x) {
  return T{2} / c10::sqrt_pi<T> * std::exp(-x * x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> erf_backward(c10::complex<T> z) {
  return T{2} / c10::sqrt_pi<T> * std::exp(-z * z);
}

} // namespace torchscience::impl::special_functions
