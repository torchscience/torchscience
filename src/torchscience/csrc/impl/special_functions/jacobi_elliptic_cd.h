#pragma once

#include <boost/math/special_functions/jacobi_elliptic.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T jacobi_elliptic_cd(T u, T k) {
  return boost::math::jacobi_cd(k, u);
}

template <typename T>
std::tuple<T, T> jacobi_elliptic_cd_backward(T u, T k) {
  // cd(u, k) = cn(u, k) / dn(u, k)
  // d/du cd(u, k) = -(1 - k^2) * sn(u, k) / dn^2(u, k)
  T sn = boost::math::jacobi_sn(k, u);
  T dn = boost::math::jacobi_dn(k, u);
  T grad_u = -(T(1) - k * k) * sn / (dn * dn);
  T grad_k = T(0);

  return std::make_tuple(grad_u, grad_k);
}

} // namespace torchscience::impl::special_functions
