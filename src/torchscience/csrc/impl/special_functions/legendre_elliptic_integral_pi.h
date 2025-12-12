#pragma once

#include <boost/math/special_functions/ellint_3.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T legendre_elliptic_integral_pi(T n, T phi, T k) {
  // Incomplete elliptic integral of the third kind Π(n; φ\k)
  // Π(n; φ\k) = integral from 0 to φ of 1 / ((1 - n*sin^2(t)) * sqrt(1 - k^2*sin^2(t))) dt
  // Boost uses ellint_3(k, n, phi) ordering
  return boost::math::ellint_3(k, n, phi);
}

template <typename T>
std::tuple<T, T, T> legendre_elliptic_integral_pi_backward(T n, T phi, T k) {
  // Partial derivatives of Π(n; φ\k)
  T sin_phi = std::sin(phi);
  T cos_phi = std::cos(phi);
  T sin2_phi = sin_phi * sin_phi;
  T k2 = k * k;
  T n_sin2 = n * sin2_phi;

  T denom1 = T(1) - n_sin2;
  T denom2 = T(1) - k2 * sin2_phi;
  T sqrt_denom2 = std::sqrt(denom2);

  // dΠ/dφ = 1 / ((1 - n*sin²φ) * sqrt(1 - k²*sin²φ))
  T grad_phi;
  if (std::abs(denom1) < T(1e-10) || std::abs(denom2) < T(1e-10)) {
    grad_phi = T(0);
  } else {
    grad_phi = T(1) / (denom1 * sqrt_denom2);
  }

  // For dΠ/dn and dΠ/dk, use numerical approximations based on relations
  // dΠ/dn involves integrals that relate to Π itself and F, E
  T Pi_val = boost::math::ellint_3(k, n, phi);
  T F_val = boost::math::ellint_1(k, phi);
  T E_val = boost::math::ellint_2(k, phi);

  // dΠ/dn: Based on the relation from DLMF 19.4
  // dΠ(n; φ|k)/dn = [E(φ|k) - (1-n)Π(n; φ|k) - k²sin(φ)cos(φ)/sqrt(1-k²sin²φ)] / [2(k²-n)(n-1)]
  T grad_n;
  T denom_n = T(2) * (k2 - n) * (n - T(1));
  if (std::abs(denom_n) < T(1e-10)) {
    grad_n = T(0);
  } else {
    T term = k2 * sin_phi * cos_phi / sqrt_denom2;
    grad_n = (E_val - (T(1) - n) * Pi_val - term) / denom_n;
  }

  // dΠ/dk: Based on the relation
  // dΠ(n; φ|k)/dk = k * [E(φ|k)/(k²-1) + Π(n; φ|k) - k²sin(φ)cos(φ)/((k²-n)*sqrt(1-k²sin²φ))] / (k²-n)
  T grad_k;
  T denom_k = k2 - n;
  if (std::abs(denom_k) < T(1e-10) || std::abs(k) < T(1e-10)) {
    grad_k = T(0);
  } else {
    T k2_minus_1 = k2 - T(1);
    T term1 = (std::abs(k2_minus_1) < T(1e-10)) ? T(0) : E_val / k2_minus_1;
    T term2 = k2 * sin_phi * cos_phi / (denom_k * sqrt_denom2);
    grad_k = k * (term1 + Pi_val - term2) / denom_k;
  }

  return std::make_tuple(grad_n, grad_phi, grad_k);
}

} // namespace torchscience::impl::special_functions
