#pragma once

#include <boost/math/special_functions/ellint_1.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
T inverse_jacobi_elliptic_cd(T x, T k) {
  // arccd(x, k) = arccn(x / sqrt(1 - (1 - k^2) * (1 - x^2)), k)
  // Simplifies to: arccd(x, k) = F(arccos(x * sqrt(1 - k^2 * (1 - x^2)) / sqrt(1 - (1-k^2)*(1-x^2))), k)
  // Alternative: arccd(x, k) = F(arccos(x / dn), k) where dn satisfies cd = x
  // For simplicity, use the elliptic integral directly
  T k_prime_sq = T(1) - k * k;
  T denom_sq = T(1) - k_prime_sq * (T(1) - x * x);
  T cn_val = x / std::sqrt(denom_sq);
  T phi = std::acos(cn_val);
  return boost::math::ellint_1(k, phi);
}

template <typename T>
std::tuple<T, T> inverse_jacobi_elliptic_cd_backward(T x, T k) {
  // d/dx arccd(x, k) = -1 / sqrt((1 - x^2)(1 - (1-k^2)*(1-x^2)))
  T k_prime_sq = T(1) - k * k;
  T grad_x = T(-1) / std::sqrt((T(1) - x * x) * (T(1) - k_prime_sq * (T(1) - x * x)));
  T grad_k = T(0);

  return std::make_tuple(grad_x, grad_k);
}

} // namespace torchscience::impl::special_functions
