#pragma once

#include <boost/math/special_functions/ellint_3.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T complete_elliptic_integral_pi(T n, T k) {
  // Complete elliptic integral of the third kind Π(n, k)
  // Π(n, k) = integral from 0 to pi/2 of 1 / ((1 - n*sin^2(t)) * sqrt(1 - k^2*sin^2(t))) dt
  return boost::math::ellint_3(k, n);
}

template <typename T>
std::tuple<T, T> complete_elliptic_integral_pi_backward(T n, T k) {
  // Partial derivatives of Π(n, k)
  // dΠ/dn and dΠ/dk are complex expressions involving E(k), K(k), and Π(n, k)
  T Pi_nk = boost::math::ellint_3(k, n);
  T K_k = boost::math::ellint_1(k);
  T E_k = boost::math::ellint_2(k);
  T k2 = k * k;
  T n2 = n * n;

  // dΠ/dn = (E(k) + (k^2 - n)*K(k)/n + (n^2 - k^2)*Π(n,k)/n) / (2*(k^2 - n)*(n - 1))
  T denom_n = T(2) * (k2 - n) * (n - T(1));
  T grad_n;
  if (std::abs(denom_n) < T(1e-10)) {
    grad_n = T(0);  // Limiting case
  } else {
    grad_n = (E_k + (k2 - n) * K_k / n + (n2 - k2) * Pi_nk / n) / denom_n;
  }

  // dΠ/dk = k * (E(k) / (k^2 - 1) + Π(n,k)) / (k^2 - n)
  T denom_k = k2 - n;
  T grad_k;
  if (std::abs(denom_k) < T(1e-10) || std::abs(k) < T(1e-10)) {
    grad_k = T(0);  // Limiting case
  } else {
    grad_k = k * (E_k / (k2 - T(1)) + Pi_nk) / denom_k;
  }

  return std::make_tuple(grad_n, grad_k);
}

} // namespace torchscience::impl::special_functions
