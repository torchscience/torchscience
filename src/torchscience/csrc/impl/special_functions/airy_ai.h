#pragma once

#include <boost/math/special_functions/airy.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T airy_ai(T x) {
  return boost::math::airy_ai(x);
}

template <typename T>
C10_HOST_DEVICE T airy_ai_backward(T x) {
  // d/dx Ai(x) = Ai'(x)
  return boost::math::airy_ai_prime(x);
}

} // namespace torchscience::impl::special_functions
