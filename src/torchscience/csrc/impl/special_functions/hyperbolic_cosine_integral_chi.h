#pragma once

#include <c10/util/numbers.h>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T hyperbolic_cosine_integral_chi(T x) {
  // Chi(x) = γ + ln(|x|) + ∫₀ˣ (cosh(t)-1)/t dt
  // For small x, use series expansion:
  // Chi(x) = γ + ln(|x|) + Σₖ₌₁^∞ x^(2k) / ((2k) * (2k)!)

  constexpr T euler_gamma = T(0.5772156649015328606065120900824024310421593359);
  T abs_x = std::abs(x);

  if (abs_x == T(0)) {
    return -std::numeric_limits<T>::infinity();
  }

  if (abs_x < T(20)) {
    // Series expansion
    T result = euler_gamma + std::log(abs_x);
    T x2 = x * x;
    T term = x2 / T(4);  // k=1: x^2 / (2 * 2!)
    result += term;

    for (int k = 2; k < 100; ++k) {
      // term_k = x^(2k) / ((2k) * (2k)!)
      // term_k / term_{k-1} = x^2 / ((2k) * (2k) * (2k-1))
      term *= x2 / (T(2 * k) * T(2 * k) * T(2 * k - 1));
      result += term;
      if (std::abs(term) < std::abs(result) * std::numeric_limits<T>::epsilon()) {
        break;
      }
    }
    return result;
  } else {
    // Asymptotic expansion for large |x|:
    // Chi(x) ≈ (e^x / (2x)) * (1 + 1/x + 2!/x^2 + ...) + (e^(-x) / (2x)) * (1 - 1/x + 2!/x^2 - ...)
    // which simplifies for large x to mainly the e^x term
    T inv_x = T(1) / abs_x;
    T exp_x = std::exp(abs_x);
    T exp_neg_x = std::exp(-abs_x);

    // Asymptotic series for Chi(x) for large x
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

    return (exp_x * sum_plus + exp_neg_x * sum_minus) * inv_x / T(2);
  }
}

template <typename T>
C10_HOST_DEVICE T hyperbolic_cosine_integral_chi_backward(T x) {
  // d/dx Chi(x) = cosh(x) / x
  return std::cosh(x) / x;
}

} // namespace torchscience::impl::special_functions
