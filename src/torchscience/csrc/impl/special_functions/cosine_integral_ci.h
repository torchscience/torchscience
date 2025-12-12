#pragma once

#include <boost/math/constants/constants.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T cosine_integral_ci(T x) {
  // Ci(x) = γ + ln|x| + Σ_{k=1}^∞ (-1)^k * x^(2k) / ((2k) * (2k)!)
  // For small x, use series expansion
  // For large x, use asymptotic expansion

  const T euler_gamma = boost::math::constants::euler<T>();
  T abs_x = std::abs(x);

  if (abs_x == T(0)) {
    return -std::numeric_limits<T>::infinity();
  }

  if (abs_x < T(40)) {
    // Series expansion: Ci(x) = γ + ln|x| + Σ_{k=1}^∞ (-1)^k * x^(2k) / ((2k) * (2k)!)
    T result = euler_gamma + std::log(abs_x);
    T x2 = x * x;
    T term = -x2 / T(4);  // First term: -x^2 / (2 * 2!) = -x^2/4
    result += term;

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
    // Asymptotic expansion for large x:
    // Ci(x) ≈ sin(x)/x * f(x) - cos(x)/x * g(x)
    // where f(x) and g(x) are auxiliary functions
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

    return std::sin(x) / x * f - std::cos(x) / x * g;
  }
}

template <typename T>
C10_HOST_DEVICE T cosine_integral_ci_backward(T x) {
  // dCi(x)/dx = cos(x)/x
  return std::cos(x) / x;
}

} // namespace torchscience::impl::special_functions
