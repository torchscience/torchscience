#pragma once

#include <boost/math/special_functions/airy.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T airy_bi_derivative(T x) {
  // Bi'(x) = derivative of Airy function Bi
  return boost::math::airy_bi_prime(x);
}

template <typename T>
C10_HOST_DEVICE T airy_bi_derivative_backward(T x) {
  // d/dx Bi'(x) = Bi''(x) = x * Bi(x) (from the Airy equation: y'' = xy)
  return x * boost::math::airy_bi(x);
}

} // namespace torchscience::impl::special_functions
