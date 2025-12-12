#pragma once

#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/jacobi_elliptic.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
T inverse_jacobi_elliptic_dn(T x, T k) {
  // arcdn(x, k) = F(arcsin(sqrt((1 - x^2) / k^2)), k)
  // This requires x in [sqrt(1-k^2), 1] for real results
  // For x = 1, result is 0; for x = sqrt(1-k^2), result is K(k)
  if (k == T(0)) {
    return T(0);  // When k=0, dn(u,0)=1 always, so inverse is 0
  }
  T sin_phi_sq = (T(1) - x * x) / (k * k);
  T sin_phi = std::sqrt(sin_phi_sq);
  T phi = std::asin(sin_phi);
  return boost::math::ellint_1(k, phi);
}

template <typename T>
std::tuple<T, T> inverse_jacobi_elliptic_dn_backward(T x, T k) {
  // d/dx arcdn(x, k) = -1 / (k * sqrt((1 - x^2)(x^2 - (1 - k^2))))
  // Valid when sqrt(1-k^2) < x < 1
  T k_prime_sq = T(1) - k * k;
  T grad_x = T(-1) / (k * std::sqrt((T(1) - x * x) * (x * x - k_prime_sq)));
  T grad_k = T(0);

  return std::make_tuple(grad_x, grad_k);
}

} // namespace torchscience::impl::special_functions
