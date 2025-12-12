#pragma once

#include <boost/math/special_functions/bernoulli.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T tangent_number_t(T n) {
  return boost::math::tangent_t2n<T>(static_cast<int>(n));
}

template <typename T>
T tangent_number_t_backward(T n) {
  // Tangent numbers are defined only at integer points
  // Gradient is zero for discrete functions
  return T(0);
}

} // namespace torchscience::impl::special_functions
