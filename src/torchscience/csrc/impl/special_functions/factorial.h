#pragma once

#include <boost/math/special_functions/factorials.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T factorial(T x) {
  return boost::math::factorial<T>(static_cast<unsigned int>(x));
}

template <typename T>
T factorial_backward(T x) {
  // d/dx(x!) = x! * psi(x+1) where psi is digamma
  // For integer arguments, we use numerical differentiation
  T eps = T(0.5);
  T f_plus = boost::math::factorial<T>(static_cast<unsigned int>(x + eps));
  T f_minus = boost::math::factorial<T>(static_cast<unsigned int>(x));
  return (f_plus - f_minus) / eps;
}

} // namespace torchscience::impl::special_functions
