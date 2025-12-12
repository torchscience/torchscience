#pragma once

#include <boost/math/special_functions/ellint_2.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <cmath>
#include <tuple>

template <typename T>
T incomplete_elliptic_integral_e(T phi, T k) {
  // Incomplete elliptic integral of the second kind E(phi, k)
  // E(phi, k) = integral from 0 to phi of sqrt(1 - k^2 * sin^2(t)) dt
  return boost::math::ellint_2(k, phi);
}

template <typename T>
std::tuple<T, T> incomplete_elliptic_integral_e_backward(T phi, T k) {
  // Partial derivatives of E(phi, k)
  // dE/dphi = sqrt(1 - k^2 * sin^2(phi))
  // dE/dk = (E(phi, k) - F(phi, k)) / k
  T sin_phi = std::sin(phi);
  T k2 = k * k;
  T sin2_phi = sin_phi * sin_phi;

  T grad_phi = std::sqrt(T(1) - k2 * sin2_phi);

  T grad_k;
  if (std::abs(k) < T(1e-10)) {
    // For k near 0, E ≈ phi and F ≈ phi
    // dE/dk ≈ 0
    grad_k = T(0);
  } else {
    T E_phi_k = boost::math::ellint_2(k, phi);
    T F_phi_k = boost::math::ellint_1(k, phi);
    grad_k = (E_phi_k - F_phi_k) / k;
  }

  return std::make_tuple(grad_phi, grad_k);
}
