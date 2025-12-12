#pragma once

#include <boost/math/special_functions/erf.hpp>
#include <boost/math/constants/constants.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T inverse_erfc(T x) {
  return boost::math::erfc_inv(x);
}

template <typename T>
C10_HOST_DEVICE T inverse_erfc_backward(T x) {
  // If y = erfc_inv(x), then erfc(y) = x
  // d/dx erfc(y) = erfc'(y) * dy/dx = 1
  // erfc'(y) = -2/sqrt(pi) * exp(-y^2)
  // dy/dx = -sqrt(pi)/2 * exp(y^2)
  T y = boost::math::erfc_inv(x);
  return -boost::math::constants::root_pi<T>() / T{2} * std::exp(y * y);
}

} // namespace torchscience::impl::special_functions
