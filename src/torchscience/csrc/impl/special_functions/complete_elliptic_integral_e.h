#pragma once

#include <boost/math/special_functions/ellint_2.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T complete_elliptic_integral_e(T k) {
  // Complete elliptic integral of the second kind E(k)
  // E(k) = integral from 0 to pi/2 of sqrt(1 - k^2 * sin^2(t)) dt
  return boost::math::ellint_2(k);
}

template <typename T>
C10_HOST_DEVICE T complete_elliptic_integral_e_backward(T k) {
  // dE(k)/dk = (E(k) - K(k)) / k
  // where K(k) is the complete elliptic integral of the first kind
  if (std::abs(k) < T(1e-10)) {
    // For k near 0, E(k) ≈ π/2 and K(k) ≈ π/2
    // dE/dk ≈ -π*k/4 for small k
    return -boost::math::constants::pi<T>() * k / T(4);
  }
  T E_k = boost::math::ellint_2(k);
  T K_k = boost::math::ellint_1(k);
  return (E_k - K_k) / k;
}

} // namespace torchscience::impl::special_functions
