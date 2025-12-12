#pragma once

#include <boost/math/special_functions/jacobi_elliptic.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T jacobi_elliptic_sn(T u, T k) {
  return boost::math::jacobi_sn(k, u);
}

template <typename T>
std::tuple<T, T> jacobi_elliptic_sn_backward(T u, T k) {
  // d/du sn(u, k) = cn(u, k) * dn(u, k)
  // d/dk sn(u, k) is more complex, we set it to 0 for now
  T cn = boost::math::jacobi_cn(k, u);
  T dn = boost::math::jacobi_dn(k, u);
  T grad_u = cn * dn;
  T grad_k = T(0);

  return std::make_tuple(grad_u, grad_k);
}

} // namespace torchscience::impl::special_functions
