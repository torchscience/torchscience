#pragma once

#include <cmath>
#include <limits>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T sine_integral_si(T x) {
  // Si(x) = integral from 0 to x of sin(t)/t dt
  // For small x, use series: Si(x) = x - x³/18 + x⁵/600 - x⁷/35280 + ...
  // For large x, use asymptotic expansion

  T abs_x = std::abs(x);
  T sign = (x >= T(0)) ? T(1) : T(-1);

  if (abs_x == T(0)) {
    return T(0);
  }

  if (abs_x < T(4)) {
    // Series expansion: Si(x) = Σ_{n=0}^∞ (-1)^n * x^(2n+1) / ((2n+1) * (2n+1)!)
    T x2 = x * x;
    T term = x;
    T result = term;

    for (int n = 1; n < 50; ++n) {
      term *= -x2 / (T(2 * n) * T(2 * n + 1));
      T contribution = term / T(2 * n + 1);
      result += contribution;
      if (std::abs(contribution) < std::abs(result) * std::numeric_limits<T>::epsilon()) {
        break;
      }
    }
    return result;
  } else {
    // Asymptotic expansion for large |x|:
    // Si(x) ≈ π/2 - f(x)*cos(x) - g(x)*sin(x)
    // where f(x) = 1/x - 2!/x³ + 4!/x⁵ - ...
    //       g(x) = 1/x² - 3!/x⁴ + 5!/x⁶ - ...
    T f = T(0);
    T g = T(0);
    T term_f = T(1) / abs_x;
    T term_g = T(1) / (abs_x * abs_x);
    f = term_f;
    g = term_g;

    for (int k = 1; k < 15; ++k) {
      term_f *= -T(2 * k - 1) * T(2 * k) / (abs_x * abs_x);
      term_g *= -T(2 * k) * T(2 * k + 1) / (abs_x * abs_x);
      T old_f = f;
      T old_g = g;
      f += term_f;
      g += term_g;
      if (f == old_f && g == old_g) {
        break;
      }
    }

    T pi_2 = T(3.14159265358979323846264338327950288) / T(2);
    T result = pi_2 - f * std::cos(abs_x) - g * std::sin(abs_x);
    return sign * result;
  }
}

template <typename T>
C10_HOST_DEVICE T sine_integral_si_backward(T x) {
  // d/dx Si(x) = sin(x)/x = sinc(x) (unnormalized)
  if (x == T(0)) {
    return T(1);
  }
  return std::sin(x) / x;
}

} // namespace torchscience::impl::special_functions
