#pragma once

#include <boost/math/special_functions/cosint.hpp>
#include <boost/math/constants/constants.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T cosine_integral_cin(T x) {
  // Cin(x) = integral from 0 to x of (1 - cos(t))/t dt
  // Cin(x) = gamma + ln(x) - Ci(x) where gamma is Euler's constant
  // Using the relation: Cin(x) = gamma + ln(|x|) - Ci(x)
  T gamma = boost::math::constants::euler<T>();
  T ci = boost::math::cosint(x);
  return gamma + std::log(std::abs(x)) - ci;
}

template <typename T>
C10_HOST_DEVICE T cosine_integral_cin_backward(T x) {
  // d/dx Cin(x) = (1 - cos(x))/x
  if (x == T(0)) {
    return T(0);
  }
  return (T(1) - std::cos(x)) / x;
}

} // namespace torchscience::impl::special_functions
