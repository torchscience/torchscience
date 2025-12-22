#pragma once

/*
 * Pentagamma Function psi'''(x) = d^4/dx^4 ln(Gamma(x))
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    The pentagamma function is the third derivative of the digamma function:
 *        psi'''(x) = d^3/dx^3 psi(x) = d^4/dx^4 ln(Gamma(x))
 *
 * 2. RECURRENCE RELATION:
 *    psi'''(x+1) = psi'''(x) - 6/x^4
 *
 * 3. REFLECTION FORMULA:
 *    psi'''(1-x) + psi'''(x) = -pi * d^3/dx^3 cot(pi*x)
 *                            = 2*pi^4 * (1 + 2*cos^2(pi*x)) / sin^4(pi*x)
 *
 * 4. ASYMPTOTIC EXPANSION:
 *    For large x:
 *        psi'''(x) ~ -2/x^3 - 3/x^4 - 2/x^5 + sum_{k=1}^inf (2k+2)(2k+1) B_{2k} / x^(2k+3)
 *
 *    where B_{2k} are Bernoulli numbers.
 *
 * 5. SPECIAL VALUES:
 *    - psi'''(1) = 6*zeta(4) = pi^4/15 = 6.494
 *    - psi'''(x) has poles at x = 0, -1, -2, -3, ...
 *
 * 6. IMPLEMENTATION:
 *    - Uses asymptotic expansion for large x (x >= 7)
 *    - Uses recurrence relation for small x
 *    - Uses reflection formula for negative x
 *
 * 7. DTYPE SUPPORT:
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
#include "cos_pi.h"
#include "type_traits.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Constants for pentagamma computation
// ============================================================================

constexpr double kPentagammaPi = 3.14159265358979323846;

// Bernoulli numbers B_{2k} for asymptotic expansions
// Coefficients are (2k+2)(2k+1) * B_{2k}
constexpr double kPentagammaC1 = 12.0 * (1.0 / 6.0);    // 4*3 * B_2 = 2
constexpr double kPentagammaC2 = 30.0 * (-1.0 / 30.0);  // 6*5 * B_4 = -1
constexpr double kPentagammaC3 = 56.0 * (1.0 / 42.0);   // 8*7 * B_6 = 4/3
constexpr double kPentagammaC4 = 90.0 * (-1.0 / 30.0);  // 10*9 * B_8 = -3
constexpr double kPentagammaC5 = 132.0 * (5.0 / 66.0);  // 12*11 * B_10 = 10

// ============================================================================
// Unified asymptotic expansion for pentagamma (works for real and complex)
// ============================================================================

/**
 * Computes the asymptotic expansion of pentagamma for large |x|:
 *   psi'''(x) ~ -2/x^3 - 3/x^4 - 2/x^5 + sum_{k=1}^{5} (2k+2)(2k+1)*B_{2k}/x^{2k+3}
 *
 * This template works for both real and complex types.
 *
 * @param x Input value with Re(x) >= 7
 * @return Asymptotic approximation of psi'''(x)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T pentagamma_asymptotic(T x) {
  using real_t = scalar_value_t<T>;

  // inv_x = 1/x, inv_x2 = 1/x^2
  T inv_x = make_scalar_for(x, real_t(1)) / x;
  T inv_x2 = inv_x * inv_x;
  T inv_x3 = inv_x2 * inv_x;
  T inv_x4 = inv_x2 * inv_x2;
  T inv_x5 = inv_x4 * inv_x;

  // Horner's method for the polynomial in 1/x^2
  T poly = make_scalar_for(x, real_t(kPentagammaC5));
  poly = make_scalar_for(x, real_t(kPentagammaC4)) + inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kPentagammaC3)) + inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kPentagammaC2)) + inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kPentagammaC1)) + inv_x2 * poly;

  // psi'''(x) = -2/x^3 - 3/x^4 - 2/x^5 + poly/x^5
  return -make_scalar_for(x, real_t(2)) * inv_x3 - make_scalar_for(x, real_t(3)) * inv_x4 - make_scalar_for(x, real_t(2)) * inv_x5 + inv_x5 * poly;
}

// ============================================================================
// Pentagamma function implementation
// ============================================================================

/**
 * Pentagamma function for real types using asymptotic expansion and recurrence.
 *
 * For small x, uses recurrence: psi'''(x+1) = psi'''(x) - 6/x^4
 *   Rearranged: psi'''(x) = psi'''(x+1) + 6/x^4
 *
 * For negative x, uses reflection formula:
 *   psi'''(1-x) + psi'''(x) = 2*pi^4 * (1 + 2*cos^2(pi*x)) / sin^4(pi*x)
 *   So: psi'''(x) = 2*pi^4 * (1 + 2*cos^2(pi*x)) / sin^4(pi*x) - psi'''(1-x)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<scalar_t>::value, scalar_t>
pentagamma(scalar_t x) {
  using std::floor;

  using real_t = scalar_value_t<scalar_t>;
  const real_t pi = real_t(kPentagammaPi);
  const real_t pi4 = pi * pi * pi * pi;

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
    scalar_t cos_pi_x = cos_pi(x);
    scalar_t sin_pi_x2 = sin_pi_x * sin_pi_x;
    scalar_t sin_pi_x4 = sin_pi_x2 * sin_pi_x2;
    return scalar_t(2) * pi4 * (scalar_t(1) + scalar_t(2) * cos_pi_x * cos_pi_x) / sin_pi_x4 - pentagamma(scalar_t(1) - x);
  }

  // Recurrence: reduce x to x >= 7 where asymptotic expansion is accurate
  scalar_t output = scalar_t(0);
  while (x < scalar_t(7)) {
    scalar_t x2 = x * x;
    output += scalar_t(6) / (x2 * x2);
    x += scalar_t(1);
  }

  return output + pentagamma_asymptotic(x);
}

/**
 * Complex pentagamma function using the same asymptotic expansion.
 *
 * The pentagamma function has poles at non-positive integers (0, -1, -2, ...).
 * For complex numbers at these poles, we return NaN.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> pentagamma(c10::complex<T> x) {
  const T pi = T(kPentagammaPi);
  const T pi4 = pi * pi * pi * pi;

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
    c10::complex<T> cos_pi_x = cos_pi(x);
    c10::complex<T> sin_pi_x2 = sin_pi_x * sin_pi_x;
    c10::complex<T> sin_pi_x4 = sin_pi_x2 * sin_pi_x2;
    return c10::complex<T>(T(2) * pi4, T(0)) * (c10::complex<T>(T(1), T(0)) + c10::complex<T>(T(2), T(0)) * cos_pi_x * cos_pi_x) / sin_pi_x4 - pentagamma(c10::complex<T>(T(1), T(0)) - x);
  }

  // Recurrence: reduce x to Re(x) >= 7 where asymptotic expansion is accurate
  c10::complex<T> output(T(0), T(0));
  while (x.real() < T(7)) {
    c10::complex<T> x2 = x * x;
    output = output + c10::complex<T>(T(6), T(0)) / (x2 * x2);
    x = x + c10::complex<T>(T(1), T(0));
  }

  return output + pentagamma_asymptotic(x);
}

}  // namespace torchscience::impl::special_functions
