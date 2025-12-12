#pragma once

#include <boost/math/special_functions/ellint_1.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
T inverse_jacobi_elliptic_sd(T x, T k) {
  // arcsd(x, k) = arcsn(x / sqrt(1 + (1 - k^2) * x^2), k)
  // where arcsn is the inverse of sn
  T k_prime_sq = T(1) - k * k;
  T denom = std::sqrt(T(1) + k_prime_sq * x * x);
  T sn_val = x / denom;
  T phi = std::asin(sn_val);
  return boost::math::ellint_1(k, phi);
}

template <typename T>
std::tuple<T, T> inverse_jacobi_elliptic_sd_backward(T x, T k) {
  // d/dx arcsd(x, k) = 1 / sqrt((1 + (1-k^2)*x^2)(1 + x^2))
  T k_prime_sq = T(1) - k * k;
  T grad_x = T(1) / std::sqrt((T(1) + k_prime_sq * x * x) * (T(1) + x * x));
  T grad_k = T(0);

  return std::make_tuple(grad_x, grad_k);
}

} // namespace torchscience::impl::special_functions
