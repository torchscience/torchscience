#pragma once

#include <boost/math/special_functions/sinint.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T sine_integral_si(T x) {
  // Si(x) = integral from 0 to x of sin(t)/t dt
  return boost::math::sinint(x);
}

template <typename T>
C10_HOST_DEVICE T sine_integral_si_backward(T x) {
  // d/dx Si(x) = sin(x)/x = sinc(x) (unnormalized)
  if (x == T(0)) {
    return T(1);
  }
  return std::sin(x) / x;
}

} // namespace torchscience::impl::special_functions
