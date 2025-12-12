#pragma once

#include <boost/math/special_functions/ellint_d.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T complete_legendre_elliptic_integral_d(T k) {
  // Complete Legendre elliptic integral D(k)
  // D(k) = (K(k) - E(k)) / k^2
  // where K(k) and E(k) are the complete elliptic integrals of the first and second kind
  return boost::math::ellint_d(k);
}

template <typename T>
C10_HOST_DEVICE T complete_legendre_elliptic_integral_d_backward(T k) {
  // dD(k)/dk
  // D(k) = (K(k) - E(k)) / k^2
  // Using quotient rule and derivatives of K and E
  if (std::abs(k) < T(1e-10)) {
    // For k near 0, D(k) ≈ π/4 * k
    // dD/dk ≈ π/4
    return boost::math::constants::pi<T>() / T(4);
  }
  T K_k = boost::math::ellint_1(k);
  T E_k = boost::math::ellint_2(k);
  T D_k = boost::math::ellint_d(k);
  T k2 = k * k;

  // dK/dk = (E(k) - (1-k^2)*K(k)) / (k*(1-k^2))
  // dE/dk = (E(k) - K(k)) / k
  // dD/dk = (K(k) - E(k) / (1-k^2)) / (k * (1 - k^2))
  T one_minus_k2 = T(1) - k2;
  return (K_k - E_k / one_minus_k2) / (k * one_minus_k2);
}

} // namespace torchscience::impl::special_functions
