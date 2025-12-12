#pragma once

#include <boost/math/special_functions/factorials.hpp>
#include <cmath>
#include <tuple>

template <typename T>
T rising_factorial(T x, T n) {
  return boost::math::rising_factorial(x, static_cast<int>(n));
}

template <typename T>
std::tuple<T, T> rising_factorial_backward(T x, T n) {
  // Numerical differentiation for the backward pass
  T eps = std::sqrt(std::numeric_limits<T>::epsilon());

  // Gradient with respect to x
  T f_x_plus = boost::math::rising_factorial(x + eps, static_cast<int>(n));
  T f_x_minus = boost::math::rising_factorial(x - eps, static_cast<int>(n));
  T grad_x = (f_x_plus - f_x_minus) / (T(2) * eps);

  // Gradient with respect to n (discrete, so we use forward difference)
  T grad_n = T(0);  // n is typically an integer, gradient is not well-defined

  return std::make_tuple(grad_x, grad_n);
}
