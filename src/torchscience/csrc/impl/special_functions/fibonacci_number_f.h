#pragma once

#include <boost/math/special_functions/fibonacci.hpp>

template <typename T>
T fibonacci_number_f(T n) {
  return static_cast<T>(boost::math::fibonacci<T>(static_cast<unsigned int>(n)));
}

template <typename T>
T fibonacci_number_f_backward(T n) {
  // Fibonacci numbers are defined only at integer points
  // Gradient is zero for discrete functions
  return T(0);
}
