#pragma once

#include <boost/math/special_functions/factorials.hpp>

template <typename T>
T double_factorial(T x) {
  return boost::math::double_factorial<T>(static_cast<unsigned int>(x));
}

template <typename T>
T double_factorial_backward(T x) {
  // Numerical differentiation for the backward pass
  T eps = T(0.5);
  T f_plus = boost::math::double_factorial<T>(static_cast<unsigned int>(x + eps));
  T f_minus = boost::math::double_factorial<T>(static_cast<unsigned int>(x));
  return (f_plus - f_minus) / eps;
}
