#pragma once

#include <boost/math/special_functions/expint.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T exponential_integral_e_1(T x) {
  // E_1(x) = integral from x to infinity of exp(-t)/t dt
  return boost::math::expint(1, x);
}

template <typename T>
C10_HOST_DEVICE T exponential_integral_e_1_backward(T x) {
  // d/dx E_1(x) = -exp(-x)/x
  return -std::exp(-x) / x;
}

} // namespace torchscience::impl::special_functions
