#pragma once

#include <boost/math/special_functions/jacobi_elliptic.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T jacobi_elliptic_dn(T u, T k) {
  return boost::math::jacobi_dn(k, u);
}

template <typename T>
std::tuple<T, T> jacobi_elliptic_dn_backward(T u, T k) {
  // d/du dn(u, k) = -k^2 * sn(u, k) * cn(u, k)
  // d/dk dn(u, k) is more complex, we set it to 0 for now
  T sn = boost::math::jacobi_sn(k, u);
  T cn = boost::math::jacobi_cn(k, u);
  T grad_u = -k * k * sn * cn;
  T grad_k = T(0);

  return std::make_tuple(grad_u, grad_k);
}

} // namespace torchscience::impl::special_functions
