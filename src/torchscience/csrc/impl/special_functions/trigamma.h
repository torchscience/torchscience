#pragma once

/*
 * Trigamma Function psi'(x) = d^2/dx^2 ln(Gamma(x))
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    The trigamma function is the derivative of the digamma function:
 *        psi'(x) = d/dx psi(x) = d^2/dx^2 ln(Gamma(x))
 *
 * 2. SPECIAL VALUES:
 *    - psi'(1) = pi^2/6
 *    - psi'(n) = pi^2/6 - sum_{k=1}^{n-1} 1/k^2 for positive integers n
 *    - psi'(x) has poles at x = 0, -1, -2, -3, ...
 *
 * 3. IMPLEMENTATION:
 *    - Uses asymptotic expansion for large x (x >= 6)
 *    - Uses recurrence relation for small x: psi'(x+1) = psi'(x) - 1/x^2
 *    - Uses reflection formula for negative x: psi'(1-x) + psi'(x) = pi^2/sin^2(pi*x)
 *
 * 4. DTYPE SUPPORT:
 *    - Supports float16, bfloat16, float32, float64
 *    - Supports complex64, complex128
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <type_traits>

#include "is_nonpositive_integer.h"
#include "sin_pi.h"
#include "type_traits.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Constants for trigamma computation
// ============================================================================

constexpr double kTrigammaPi = 3.14159265358979323846;

// Asymptotic expansion coefficients (Bernoulli numbers)
// B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30, B_10 = 5/66
constexpr double kTrigammaB2 = 1.0 / 6.0;
constexpr double kTrigammaB4 = 1.0 / 30.0;
constexpr double kTrigammaB6 = 1.0 / 42.0;
constexpr double kTrigammaB8 = 1.0 / 30.0;
constexpr double kTrigammaB10 = 5.0 / 66.0;

// ============================================================================
// Unified asymptotic expansion for trigamma (works for real and complex)
// ============================================================================

/**
 * Computes the asymptotic expansion of trigamma for large |x|:
 *   psi'(x) ~ 1/x + 1/(2x^2) + sum_{k=1}^{5} B_{2k}/x^{2k+1}
 *
 * This template works for both real and complex types.
 *
 * @param x Input value with Re(x) >= 6
 * @return Asymptotic approximation of psi'(x)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T trigamma_asymptotic(T x) {
  using real_t = scalar_value_t<T>;

  // inv_x = 1/x, inv_x2 = 1/x^2
  T inv_x = make_scalar_for(x, real_t(1)) / x;
  T inv_x2 = inv_x * inv_x;

  // Horner's method for the polynomial in 1/x^2:
  // B_2/x^3 - B_4/x^5 + B_6/x^7 - B_8/x^9 + B_10/x^11
  // = (1/x^3) * (B_2 - (1/x^2) * (B_4 + (1/x^2) * (-B_6 + ...)))
  T poly = make_scalar_for(x, real_t(kTrigammaB10));
  poly = make_scalar_for(x, real_t(kTrigammaB8)) - inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kTrigammaB6)) - inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kTrigammaB4)) - inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kTrigammaB2)) - inv_x2 * poly;

  // psi'(x) = 1/x + 1/(2x^2) + poly/x^3
  T inv_x3 = inv_x2 * inv_x;
  return inv_x + inv_x2 / make_scalar_for(x, real_t(2)) + inv_x3 * poly;
}

// ============================================================================
// Trigamma function implementation
// ============================================================================

/**
 * Trigamma function for real types using asymptotic expansion and recurrence.
 *
 * Uses the asymptotic expansion for large x:
 *   psi'(x) ~ 1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7) - ...
 *
 * For small x, uses recurrence: psi'(x+1) = psi'(x) - 1/x^2
 * For negative x, uses reflection: psi'(1-x) + psi'(x) = pi^2/sin^2(pi*x)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<scalar_t>::value, scalar_t>
trigamma(scalar_t x) {
  using std::floor;

  using real_t = scalar_value_t<scalar_t>;
  const real_t pi = real_t(kTrigammaPi);

  // Handle NaN
  if (x != x) {
    return x;
  }

  // For negative x, use reflection formula
  if (x <= scalar_t(0)) {
    if (x == floor(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    scalar_t sin_pi_x = sin_pi(x);
    return pi * pi / (sin_pi_x * sin_pi_x) - trigamma(scalar_t(1) - x);
  }

  // Recurrence: reduce x to x >= 6 where asymptotic expansion is accurate
  scalar_t output = scalar_t(0);
  while (x < scalar_t(6)) {
    output += scalar_t(1) / (x * x);
    x += scalar_t(1);
  }

  return output + trigamma_asymptotic(x);
}

/**
 * Complex trigamma function using the same asymptotic expansion.
 *
 * The trigamma function has poles at non-positive integers (0, -1, -2, ...).
 * For complex numbers at these poles, we return NaN.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> trigamma(c10::complex<T> x) {
  const T pi = T(kTrigammaPi);

  // Check for poles at non-positive integers (z = 0, -1, -2, ...)
  if (is_nonpositive_integer(x)) {
    return c10::complex<T>(
      std::numeric_limits<T>::quiet_NaN(),
      std::numeric_limits<T>::quiet_NaN()
    );
  }

  // For Re(x) < 0.5, use reflection formula
  if (x.real() < T(0.5)) {
    c10::complex<T> sin_pi_x = sin_pi(x);
    return c10::complex<T>(pi * pi, T(0)) / (sin_pi_x * sin_pi_x) - trigamma(c10::complex<T>(T(1), T(0)) - x);
  }

  // Recurrence: reduce x to Re(x) >= 6 where asymptotic expansion is accurate
  c10::complex<T> output(T(0), T(0));
  while (x.real() < T(6)) {
    output = output + c10::complex<T>(T(1), T(0)) / (x * x);
    x = x + c10::complex<T>(T(1), T(0));
  }

  return output + trigamma_asymptotic(x);
}

}  // namespace torchscience::impl::special_functions
