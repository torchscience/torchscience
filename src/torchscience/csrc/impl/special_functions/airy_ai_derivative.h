#pragma once

#include <boost/math/special_functions/airy.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T airy_ai_derivative(T x) {
  // Ai'(x) = derivative of Airy function Ai
  return boost::math::airy_ai_prime(x);
}

template <typename T>
C10_HOST_DEVICE T airy_ai_derivative_backward(T x) {
  // d/dx Ai'(x) = Ai''(x) = x * Ai(x) (from the Airy equation: y'' = xy)
  return x * boost::math::airy_ai(x);
}

} // namespace torchscience::impl::special_functions
