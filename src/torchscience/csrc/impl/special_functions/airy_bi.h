#pragma once

#include <boost/math/special_functions/airy.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T airy_bi(T x) {
  return boost::math::airy_bi(x);
}

template <typename T>
C10_HOST_DEVICE T airy_bi_backward(T x) {
  // d/dx Bi(x) = Bi'(x)
  return boost::math::airy_bi_prime(x);
}

} // namespace torchscience::impl::special_functions
