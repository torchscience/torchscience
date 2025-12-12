#pragma once

#include <boost/math/special_functions/bernoulli.hpp>

template <typename T>
T bernoulli_number_b(T n) {
  return boost::math::bernoulli_b2n<T>(static_cast<int>(n));
}

template <typename T>
T bernoulli_number_b_backward(T n) {
  // Bernoulli numbers are defined only at integer points
  // Gradient is zero for discrete functions
  return T(0);
}
