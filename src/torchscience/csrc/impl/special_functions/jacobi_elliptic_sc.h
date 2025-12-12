#pragma once

#include <boost/math/special_functions/jacobi_elliptic.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T jacobi_elliptic_sc(T u, T k) {
  return boost::math::jacobi_sc(k, u);
}

template <typename T>
std::tuple<T, T> jacobi_elliptic_sc_backward(T u, T k) {
  // sc(u, k) = sn(u, k) / cn(u, k)
  // d/du sc(u, k) = dn(u, k) / cn^2(u, k)
  T cn = boost::math::jacobi_cn(k, u);
  T dn = boost::math::jacobi_dn(k, u);
  T grad_u = dn / (cn * cn);
  T grad_k = T(0);

  return std::make_tuple(grad_u, grad_k);
}

} // namespace torchscience::impl::special_functions
