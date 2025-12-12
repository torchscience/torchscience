#pragma once

#include <boost/math/special_functions/jacobi_elliptic.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T jacobi_amplitude_am(T u, T k) {
  return boost::math::jacobi_am(k, u);
}

template <typename T>
std::tuple<T, T> jacobi_amplitude_am_backward(T u, T k) {
  // am(u, k) is defined such that sn(u, k) = sin(am(u, k))
  // d/du am(u, k) = dn(u, k)
  // d/dk am(u, k) is more complex, we set it to 0 for now
  T grad_u = boost::math::jacobi_dn(k, u);
  T grad_k = T(0);

  return std::make_tuple(grad_u, grad_k);
}

} // namespace torchscience::impl::special_functions
