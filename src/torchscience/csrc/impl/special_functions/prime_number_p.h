#pragma once

#include <boost/math/special_functions/prime.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T prime_number_p(T n) {
  return static_cast<T>(boost::math::prime(static_cast<unsigned int>(n)));
}

template <typename T>
T prime_number_p_backward(T n) {
  // Prime numbers are defined only at integer points
  // Gradient is zero for discrete functions
  return T(0);
}

} // namespace torchscience::impl::special_functions
