#pragma once

#include <boost/math/special_functions/expint.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T logarithmic_integral_li(T x) {
  // li(x) = Ei(ln(x)) where Ei is the exponential integral
  return boost::math::expint(std::log(x));
}

template <typename T>
C10_HOST_DEVICE T logarithmic_integral_li_backward(T x) {
  // d/dx li(x) = 1 / ln(x)
  return T(1) / std::log(x);
}

} // namespace torchscience::impl::special_functions
