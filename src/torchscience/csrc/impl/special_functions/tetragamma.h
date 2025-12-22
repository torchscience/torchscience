#pragma once

/*
 * Tetragamma Function psi''(x) = d^3/dx^3 ln(Gamma(x))
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    The tetragamma function is the second derivative of the digamma function:
 *        psi''(x) = d^2/dx^2 psi(x) = d^3/dx^3 ln(Gamma(x))
 *
 * 2. RECURRENCE RELATION:
 *    psi''(x+1) = psi''(x) + 2/x^3
 *
 * 3. REFLECTION FORMULA:
 *    psi''(1-x) - psi''(x) = 2*pi^3 * cos(pi*x) / sin^3(pi*x)
 *
 * 4. ASYMPTOTIC EXPANSION:
 *    For large x:
 *        psi''(x) ~ -1/x^2 - 1/x^3 - 1/(2x^4) + sum_{k=1}^inf (2k+1) B_{2k} / x^(2k+2)
 *
 *    where B_{2k} are Bernoulli numbers.
 *
 * 5. SPECIAL VALUES:
 *    - psi''(1) = -2*zeta(3) = -2.404
 *    - psi''(x) has poles at x = 0, -1, -2, -3, ...
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
// Constants for tetragamma computation
// ============================================================================

constexpr double kTetragammaPi = 3.14159265358979323846;

// Bernoulli numbers B_{2k} for asymptotic expansions
// Coefficients are (2k+1) * B_{2k}
constexpr double kTetragammaC1 = 3.0 * (1.0 / 6.0);     // 3 * B_2 = 1/2
constexpr double kTetragammaC2 = 5.0 * (-1.0 / 30.0);   // 5 * B_4 = -1/6
constexpr double kTetragammaC3 = 7.0 * (1.0 / 42.0);    // 7 * B_6 = 1/6
constexpr double kTetragammaC4 = 9.0 * (-1.0 / 30.0);   // 9 * B_8 = -3/10
constexpr double kTetragammaC5 = 11.0 * (5.0 / 66.0);   // 11 * B_10 = 5/6

// ============================================================================
// Unified asymptotic expansion for tetragamma (works for real and complex)
// ============================================================================

/**
 * Computes the asymptotic expansion of tetragamma for large |x|:
 *   psi''(x) ~ -1/x^2 - 1/x^3 - 1/(2x^4) + sum_{k=1}^{5} (2k+1)*B_{2k}/x^{2k+2}
 *
 * This template works for both real and complex types.
 *
 * @param x Input value with Re(x) >= 7
 * @return Asymptotic approximation of psi''(x)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T tetragamma_asymptotic(T x) {
  using real_t = scalar_value_t<T>;

  // inv_x = 1/x, inv_x2 = 1/x^2
  T inv_x = make_scalar_for(x, real_t(1)) / x;
  T inv_x2 = inv_x * inv_x;
  T inv_x3 = inv_x2 * inv_x;
  T inv_x4 = inv_x2 * inv_x2;

  // Horner's method for the polynomial in 1/x^2
  T poly = make_scalar_for(x, real_t(kTetragammaC5));
  poly = make_scalar_for(x, real_t(kTetragammaC4)) + inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kTetragammaC3)) + inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kTetragammaC2)) + inv_x2 * poly;
  poly = make_scalar_for(x, real_t(kTetragammaC1)) + inv_x2 * poly;

  // psi''(x) = -1/x^2 - 1/x^3 - 1/(2x^4) + poly/x^4
  return -inv_x2 - inv_x3 - inv_x4 / make_scalar_for(x, real_t(2)) + inv_x4 * poly;
}

// ============================================================================
// Tetragamma function implementation
// ============================================================================

/**
 * Tetragamma function for real types using asymptotic expansion and recurrence.
 *
 * For small x, uses recurrence: psi''(x+1) = psi''(x) + 2/x^3
 *   Rearranged: psi''(x) = psi''(x+1) - 2/x^3
 *
 * For negative x, uses reflection formula:
 *   psi''(1-x) - psi''(x) = 2*pi^3 * cos(pi*x) / sin^3(pi*x)
 *   So: psi''(x) = psi''(1-x) - 2*pi^3 * cos(pi*x) / sin^3(pi*x)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<scalar_t>::value, scalar_t>
tetragamma(scalar_t x) {
  using std::floor;

  using real_t = scalar_value_t<scalar_t>;
  const real_t pi = real_t(kTetragammaPi);
  const real_t pi3 = pi * pi * pi;

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
    scalar_t sin_pi_x3 = sin_pi_x * sin_pi_x * sin_pi_x;
    return tetragamma(scalar_t(1) - x) - scalar_t(2) * pi3 * cos_pi(x) / sin_pi_x3;
  }

  // Recurrence: reduce x to x >= 7 where asymptotic expansion is accurate
  scalar_t output = scalar_t(0);
  while (x < scalar_t(7)) {
    output -= scalar_t(2) / (x * x * x);
    x += scalar_t(1);
  }

  return output + tetragamma_asymptotic(x);
}

/**
 * Complex tetragamma function using the same asymptotic expansion.
 *
 * The tetragamma function has poles at non-positive integers (0, -1, -2, ...).
 * For complex numbers at these poles, we return NaN.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> tetragamma(c10::complex<T> x) {
  const T pi = T(kTetragammaPi);
  const T pi3 = pi * pi * pi;

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
    c10::complex<T> sin_pi_x3 = sin_pi_x * sin_pi_x * sin_pi_x;
    return tetragamma(c10::complex<T>(T(1), T(0)) - x) - c10::complex<T>(T(2) * pi3, T(0)) * cos_pi(x) / sin_pi_x3;
  }

  // Recurrence: reduce x to Re(x) >= 7 where asymptotic expansion is accurate
  c10::complex<T> output(T(0), T(0));
  while (x.real() < T(7)) {
    output = output - c10::complex<T>(T(2), T(0)) / (x * x * x);
    x = x + c10::complex<T>(T(1), T(0));
  }

  return output + tetragamma_asymptotic(x);
}

}  // namespace torchscience::impl::special_functions
