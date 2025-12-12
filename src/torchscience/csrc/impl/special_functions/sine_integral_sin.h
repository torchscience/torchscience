#pragma once

#include <boost/math/special_functions/sinint.hpp>
#include <boost/math/constants/constants.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T sine_integral_sin(T x) {
  // si(x) = Si(x) - pi/2 = -integral from x to infinity of sin(t)/t dt
  // This is the "shifted" sine integral that goes to 0 as x -> infinity
  return boost::math::sinint(x) - boost::math::constants::half_pi<T>();
}

template <typename T>
C10_HOST_DEVICE T sine_integral_sin_backward(T x) {
  // d/dx si(x) = d/dx Si(x) = sin(x)/x = sinc(x) (unnormalized)
  if (x == T(0)) {
    return T(1);
  }
  return std::sin(x) / x;
}

} // namespace torchscience::impl::special_functions
