#pragma once

#include <boost/math/special_functions/erf.hpp>
#include <boost/math/constants/constants.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T inverse_erf(T x) {
  return boost::math::erf_inv(x);
}

template <typename T>
C10_HOST_DEVICE T inverse_erf_backward(T x) {
  // If y = erf_inv(x), then erf(y) = x
  // d/dx erf(y) = erf'(y) * dy/dx = 1
  // erf'(y) = 2/sqrt(pi) * exp(-y^2)
  // dy/dx = sqrt(pi)/2 * exp(y^2)
  T y = boost::math::erf_inv(x);
  return boost::math::constants::root_pi<T>() / T{2} * std::exp(y * y);
}

} // namespace torchscience::impl::special_functions
