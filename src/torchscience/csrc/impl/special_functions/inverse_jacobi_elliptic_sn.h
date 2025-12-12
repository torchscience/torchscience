#pragma once

#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/jacobi_elliptic.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
T inverse_jacobi_elliptic_sn(T x, T k) {
  // arcsn(x, k) = F(arcsin(x), k)
  // where F is the incomplete elliptic integral of the first kind
  T phi = std::asin(x);
  return boost::math::ellint_1(k, phi);
}

template <typename T>
std::tuple<T, T> inverse_jacobi_elliptic_sn_backward(T x, T k) {
  // d/dx arcsn(x, k) = 1 / sqrt((1 - x^2)(1 - k^2*x^2))
  T grad_x = T(1) / std::sqrt((T(1) - x * x) * (T(1) - k * k * x * x));
  T grad_k = T(0);

  return std::make_tuple(grad_x, grad_k);
}

} // namespace torchscience::impl::special_functions
