#pragma once

#include <boost/math/special_functions/ellint_1.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T incomplete_elliptic_integral_f(T phi, T k) {
  // Incomplete elliptic integral of the first kind F(phi, k)
  // F(phi, k) = integral from 0 to phi of 1 / sqrt(1 - k^2 * sin^2(t)) dt
  return boost::math::ellint_1(k, phi);
}

template <typename T>
std::tuple<T, T> incomplete_elliptic_integral_f_backward(T phi, T k) {
  // Partial derivatives of F(phi, k)
  // dF/dphi = 1 / sqrt(1 - k^2 * sin^2(phi))
  // dF/dk = (E(phi, k) - (1-k^2)*F(phi, k)) / (k*(1-k^2)) - k*sin(phi)*cos(phi) / ((1-k^2)*sqrt(1-k^2*sin^2(phi)))
  T sin_phi = std::sin(phi);
  T cos_phi = std::cos(phi);
  T k2 = k * k;
  T sin2_phi = sin_phi * sin_phi;
  T denom = std::sqrt(T(1) - k2 * sin2_phi);

  T grad_phi = T(1) / denom;

  T grad_k;
  if (std::abs(k) < T(1e-10)) {
    // For k near 0, F ≈ phi
    // dF/dk ≈ 0
    grad_k = T(0);
  } else {
    T E_phi_k = boost::math::ellint_2(k, phi);
    T F_phi_k = boost::math::ellint_1(k, phi);
    T one_minus_k2 = T(1) - k2;
    grad_k = (E_phi_k - one_minus_k2 * F_phi_k) / (k * one_minus_k2) -
             k * sin_phi * cos_phi / (one_minus_k2 * denom);
  }

  return std::make_tuple(grad_phi, grad_k);
}

} // namespace torchscience::impl::special_functions
