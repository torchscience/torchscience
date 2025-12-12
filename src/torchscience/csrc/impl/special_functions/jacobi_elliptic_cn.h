#pragma once

#include <boost/math/special_functions/jacobi_elliptic.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T jacobi_elliptic_cn(T u, T k) {
  return boost::math::jacobi_cn(k, u);
}

template <typename T>
std::tuple<T, T> jacobi_elliptic_cn_backward(T u, T k) {
  // d/du cn(u, k) = -sn(u, k) * dn(u, k)
  // d/dk cn(u, k) is more complex, we set it to 0 for now
  T sn = boost::math::jacobi_sn(k, u);
  T dn = boost::math::jacobi_dn(k, u);
  T grad_u = -sn * dn;
  T grad_k = T(0);

  return std::make_tuple(grad_u, grad_k);
}

} // namespace torchscience::impl::special_functions
