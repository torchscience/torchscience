#pragma once

#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/jacobi_elliptic.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
T inverse_jacobi_elliptic_cn(T x, T k) {
  // arccn(x, k) = F(arccos(x), k)
  // where F is the incomplete elliptic integral of the first kind
  T phi = std::acos(x);
  return boost::math::ellint_1(k, phi);
}

template <typename T>
std::tuple<T, T> inverse_jacobi_elliptic_cn_backward(T x, T k) {
  // d/dx arccn(x, k) = -1 / sqrt((1 - x^2)(1 - k^2 + k^2*x^2))
  // Note: 1 - k^2 + k^2*x^2 = (1 - k^2)(1 - x^2) + x^2 when expanded differently
  // The correct form is: -1 / sqrt((1 - x^2)(k'^2 + k^2*x^2)) where k' = sqrt(1 - k^2)
  T k_prime_sq = T(1) - k * k;
  T grad_x = T(-1) / std::sqrt((T(1) - x * x) * (k_prime_sq + k * k * x * x));
  T grad_k = T(0);

  return std::make_tuple(grad_x, grad_k);
}

} // namespace torchscience::impl::special_functions
