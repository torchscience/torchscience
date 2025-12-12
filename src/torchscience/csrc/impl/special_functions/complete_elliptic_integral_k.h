#pragma once

#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T complete_elliptic_integral_k(T k) {
  // Complete elliptic integral of the first kind K(k)
  // K(k) = integral from 0 to pi/2 of 1 / sqrt(1 - k^2 * sin^2(t)) dt
  return boost::math::ellint_1(k);
}

template <typename T>
C10_HOST_DEVICE T complete_elliptic_integral_k_backward(T k) {
  // dK(k)/dk = E(k) / (k * (1 - k^2)) - K(k) / k
  // where E(k) is the complete elliptic integral of the second kind
  if (std::abs(k) < T(1e-10)) {
    // For k near 0, K(k) ≈ π/2 and E(k) ≈ π/2
    // dK/dk ≈ π*k/4 for small k
    return boost::math::constants::pi<T>() * k / T(4);
  }
  T K_k = boost::math::ellint_1(k);
  T E_k = boost::math::ellint_2(k);
  T k2 = k * k;
  return E_k / (k * (T(1) - k2)) - K_k / k;
}

} // namespace torchscience::impl::special_functions
