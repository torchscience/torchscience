#pragma once

#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T hyperbolic_sine_integral_shi(T x) {
  // Shi(x) = ∫₀ˣ sinh(t)/t dt
  // Series expansion: Shi(x) = Σₖ₌₀^∞ x^(2k+1) / ((2k+1) * (2k+1)!)

  T abs_x = std::abs(x);

  if (abs_x == T(0)) {
    return T(0);
  }

  if (abs_x < T(20)) {
    // Series expansion
    T x2 = x * x;
    T term = x;  // k=0: x / (1 * 1!)
    T result = term;

    for (int k = 1; k < 100; ++k) {
      // term_k = x^(2k+1) / ((2k+1) * (2k+1)!)
      // term_k / term_{k-1} = x^2 / ((2k+1) * (2k) * (2k+1))
      term *= x2 / (T(2 * k) * T(2 * k + 1) * T(2 * k + 1));
      result += term;
      if (std::abs(term) < std::abs(result) * std::numeric_limits<T>::epsilon()) {
        break;
      }
    }
    return result;
  } else {
    // Asymptotic expansion for large |x|:
    // Shi(x) ≈ (e^x / (2x)) * (1 + 1/x + 2!/x^2 + ...) - (e^(-x) / (2x)) * (1 - 1/x + 2!/x^2 - ...)
    // which simplifies for large x to mainly the e^x term
    T sign = (x > T(0)) ? T(1) : T(-1);
    T inv_x = T(1) / abs_x;
    T exp_x = std::exp(abs_x);
    T exp_neg_x = std::exp(-abs_x);

    // Asymptotic series
    T sum_plus = T(1);
    T sum_minus = T(1);
    T term_plus = T(1);
    T term_minus = T(1);

    for (int k = 1; k < 20; ++k) {
      term_plus *= T(k) * inv_x;
      term_minus *= -T(k) * inv_x;
      T old_sum_plus = sum_plus;
      T old_sum_minus = sum_minus;
      sum_plus += term_plus;
      sum_minus += term_minus;
      if (sum_plus == old_sum_plus && sum_minus == old_sum_minus) {
        break;
      }
    }

    return sign * (exp_x * sum_plus - exp_neg_x * sum_minus) * inv_x / T(2);
  }
}

template <typename T>
C10_HOST_DEVICE T hyperbolic_sine_integral_shi_backward(T x) {
  // d/dx Shi(x) = sinh(x) / x
  if (x == T(0)) {
    return T(1);  // limit as x -> 0 of sinh(x)/x = 1
  }
  return std::sinh(x) / x;
}

} // namespace torchscience::impl::special_functions
