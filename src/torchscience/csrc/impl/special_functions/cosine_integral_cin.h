#pragma once

#include <boost/math/constants/constants.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T cosine_integral_cin(T x) {
  // Cin(x) = integral from 0 to x of (1 - cos(t))/t dt
  // Cin(x) = gamma + ln|x| - Ci(x)
  //
  // Direct series: Cin(x) = Σ_{k=1}^∞ (-1)^(k+1) * x^(2k) / ((2k) * (2k)!)

  T abs_x = std::abs(x);

  if (abs_x == T(0)) {
    return T(0);
  }

  if (abs_x < T(40)) {
    // Series expansion
    T x2 = x * x;
    T result = x2 / T(4);  // First term: x^2 / (2 * 2!) = x^2/4
    T term = result;

    for (int k = 2; k < 100; ++k) {
      term *= -x2 / (T(2 * k - 1) * T(2 * k) * T(2 * k));
      term *= T(2 * k - 2);
      result += term;
      if (std::abs(term) < std::abs(result) * std::numeric_limits<T>::epsilon()) {
        break;
      }
    }
    return result;
  } else {
    // For large x, use: Cin(x) = gamma + ln|x| - Ci(x)
    // where Ci(x) ≈ sin(x)/x * f(x) - cos(x)/x * g(x) asymptotically
    const T euler_gamma = boost::math::constants::euler<T>();

    // Compute Ci(x) asymptotically
    T f = T(1);
    T g = T(1) / x;
    T term_f = T(1);
    T term_g = T(1) / x;
    T x2_inv = T(1) / (x * x);

    for (int k = 1; k < 20; ++k) {
      term_f *= -T(2 * k - 1) * T(2 * k) * x2_inv;
      term_g *= -T(2 * k) * T(2 * k + 1) * x2_inv;
      f += term_f;
      g += term_g;
      if (std::abs(term_f) < std::abs(f) * std::numeric_limits<T>::epsilon() &&
          std::abs(term_g) < std::abs(g) * std::numeric_limits<T>::epsilon()) {
        break;
      }
    }

    T ci = std::sin(x) / x * f - std::cos(x) / x * g;
    return euler_gamma + std::log(abs_x) - ci;
  }
}

template <typename T>
C10_HOST_DEVICE T cosine_integral_cin_backward(T x) {
  // d/dx Cin(x) = (1 - cos(x))/x
  if (x == T(0)) {
    return T(0);
  }
  return (T(1) - std::cos(x)) / x;
}

} // namespace torchscience::impl::special_functions
