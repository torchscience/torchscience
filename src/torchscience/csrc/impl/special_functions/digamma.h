#pragma once

/*
 * Digamma Function psi(x) = d/dx ln(Gamma(x)) = Gamma'(x)/Gamma(x)
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    The digamma function is the logarithmic derivative of the gamma function:
 *        psi(x) = d/dx ln(Gamma(x)) = Gamma'(x)/Gamma(x)
 *
 * 2. SPECIAL VALUES:
 *    - psi(1) = -gamma (negative Euler-Mascheroni constant = -0.5772)
 *    - psi(n) = -gamma + sum_{k=1}^{n-1} 1/k for positive integers n
 *    - psi(x) has poles at x = 0, -1, -2, -3, ...
 *
 * 3. IMPLEMENTATION:
 *    - Uses asymptotic expansion for large x (x >= 6)
 *    - Uses recurrence relation for small x: psi(x+1) = psi(x) + 1/x
 *    - Uses reflection formula for negative x: psi(1-x) - psi(x) = pi*cot(pi*x)
 *
 * 4. DTYPE SUPPORT:
 *    - Supports float16, bfloat16, float32, float64
 *    - Supports complex64, complex128
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "is_nonpositive_integer.h"
#include "sin_pi.h"
#include "cos_pi.h"
#include "type_traits.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Constants for digamma computation
// ============================================================================

// pi constant
constexpr double kDigammaPi = 3.14159265358979323846;

// Asymptotic expansion coefficients (Bernoulli numbers / 2k)
// B_2 / 2 = 1/12, B_4 / 4 = -1/120, B_6 / 6 = 1/252, B_8 / 8 = -1/240, B_10 / 10 = 5/660
constexpr double kDigammaB2_over_2 = 1.0 / 12.0;
constexpr double kDigammaB4_over_4 = 1.0 / 120.0;
constexpr double kDigammaB6_over_6 = 1.0 / 252.0;
constexpr double kDigammaB8_over_8 = 1.0 / 240.0;
constexpr double kDigammaB10_over_10 = 5.0 / 660.0;

// ============================================================================
// Unified asymptotic expansion for digamma (works for real and complex)
// ============================================================================

/**
 * Computes the asymptotic expansion of digamma for large |x|:
 *   psi(x) ~ ln(x) - 1/(2x) - sum_{k=1}^{5} B_{2k}/(2k * x^{2k})
 *
 * This template works for both real and complex types.
 *
 * @param x Input value with Re(x) >= 6
 * @return Asymptotic approximation of psi(x)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T digamma_asymptotic(T x) {
  using std::log;
  using real_t = typename c10::scalar_value_type<T>::type;

  // inv_x = 1/x, inv_x2 = 1/x^2
  T inv_x = make_scalar_for(x, real_t(1)) / x;
  T inv_x2 = inv_x * inv_x;

  // Horner's method for the polynomial in 1/x^2:
  // -B_2/(2*x^2) + B_4/(4*x^4) - B_6/(6*x^6) + B_8/(8*x^8) - B_10/(10*x^10)
  // = (1/x^2) * (-B_2/2 + (1/x^2) * (B_4/4 + (1/x^2) * (-B_6/6 + ...)))
  T poly = make_scalar_for(x, real_t(kDigammaB10_over_10));
  poly = make_scalar_for(x, real_t(kDigammaB8_over_8)) - inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kDigammaB6_over_6)) - inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kDigammaB4_over_4)) - inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kDigammaB2_over_2)) - inv_x2 * poly;

  // psi(x) = ln(x) - 1/(2x) - poly/x^2
  return log(x) - inv_x / make_scalar_for(x, real_t(2)) - inv_x2 * poly;
}

// ============================================================================
// Digamma function implementation
// ============================================================================

/**
 * Digamma function for real types using asymptotic expansion and recurrence.
 *
 * Uses the asymptotic expansion for large x:
 *   psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
 *
 * For small x, uses recurrence: psi(x+1) = psi(x) + 1/x
 * For negative x, uses reflection: psi(1-x) = psi(x) + pi*cot(pi*x)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<scalar_t>::value, scalar_t>
digamma(scalar_t x) {
  using std::floor;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t pi = real_t(kDigammaPi);

  // Handle NaN
  if (x != x) {
    return x;
  }

  // For negative x, use reflection formula: psi(1-x) - psi(x) = pi*cot(pi*x)
  // So: psi(x) = psi(1-x) - pi*cot(pi*x)
  if (x <= scalar_t(0)) {
    // Check if x is a non-positive integer (pole)
    if (x == floor(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    // Reflection: psi(x) = psi(1-x) - pi*cot(pi*x)
    // Use range-reduced sin_pi/cos_pi for numerical stability with large negative x
    scalar_t cot_pi_x = cos_pi(x) / sin_pi(x);
    return digamma(scalar_t(1) - x) - pi * cot_pi_x;
  }

  // Recurrence: reduce x to x >= 6 where asymptotic expansion is accurate
  scalar_t output = scalar_t(0);
  while (x < scalar_t(6)) {
    output -= scalar_t(1) / x;
    x += scalar_t(1);
  }

  return output + digamma_asymptotic(x);
}

/**
 * Complex digamma function using the same asymptotic expansion.
 *
 * The digamma function has poles at non-positive integers (0, -1, -2, ...).
 * For complex numbers at these poles, we return NaN.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> digamma(c10::complex<T> x) {
  const T pi = T(kDigammaPi);

  // Check for poles at non-positive integers (z = 0, -1, -2, ...)
  if (is_nonpositive_integer(x)) {
    return c10::complex<T>(
      std::numeric_limits<T>::quiet_NaN(),
      std::numeric_limits<T>::quiet_NaN()
    );
  }

  // For Re(x) < 0.5, use reflection formula
  if (x.real() < T(0.5)) {
    c10::complex<T> cot_pi_x = cos_pi(x) / sin_pi(x);
    return digamma(c10::complex<T>(T(1), T(0)) - x) - c10::complex<T>(pi, T(0)) * cot_pi_x;
  }

  // Recurrence: reduce x to Re(x) >= 6 where asymptotic expansion is accurate
  c10::complex<T> output(T(0), T(0));
  while (x.real() < T(6)) {
    output -= c10::complex<T>(T(1), T(0)) / x;
    x += c10::complex<T>(T(1), T(0));
  }

  return output + digamma_asymptotic(x);
}

}  // namespace torchscience::impl::special_functions
