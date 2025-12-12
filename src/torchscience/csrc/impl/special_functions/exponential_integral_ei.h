#pragma once

#include <boost/math/special_functions/expint.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T exponential_integral_ei(T x) {
  return boost::math::expint(x);
}

template <typename T>
C10_HOST_DEVICE T exponential_integral_ei_backward(T x) {
  // dEi(x)/dx = e^x / x
  return std::exp(x) / x;
}

} // namespace torchscience::impl::special_functions
