#pragma once

#include <boost/math/special_functions/ellint_d.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T incomplete_legendre_elliptic_integral_d(T phi, T k) {
  // Incomplete Legendre elliptic integral D(phi, k)
  // D(phi, k) = (F(phi, k) - E(phi, k)) / k^2
  return boost::math::ellint_d(k, phi);
}

template <typename T>
std::tuple<T, T> incomplete_legendre_elliptic_integral_d_backward(T phi, T k) {
  // Partial derivatives of D(phi, k)
  // D(phi, k) = (F(phi, k) - E(phi, k)) / k^2
  T sin_phi = std::sin(phi);
  T k2 = k * k;
  T sin2_phi = sin_phi * sin_phi;
  T denom = std::sqrt(T(1) - k2 * sin2_phi);

  // dD/dphi = sin^2(phi) / sqrt(1 - k^2 * sin^2(phi))
  T grad_phi = sin2_phi / denom;

  T grad_k;
  if (std::abs(k) < T(1e-10)) {
    // For k near 0, D ≈ 0
    grad_k = T(0);
  } else {
    T D_phi_k = boost::math::ellint_d(k, phi);
    T F_phi_k = boost::math::ellint_1(k, phi);
    T E_phi_k = boost::math::ellint_2(k, phi);
    T one_minus_k2 = T(1) - k2;

    // dD/dk = (F - E/(1-k^2)) / (k * (1-k^2)) - 2*D/k
    grad_k = (F_phi_k - E_phi_k / one_minus_k2) / (k * one_minus_k2) - T(2) * D_phi_k / k;
  }

  return std::make_tuple(grad_phi, grad_k);
}

} // namespace torchscience::impl::special_functions
