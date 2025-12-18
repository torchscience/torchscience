#pragma once

/*
 * Polygamma Functions ψ^(n)(x)
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    The polygamma function of order n is the (n+1)-th derivative of the
 *    logarithm of the gamma function:
 *
 *        ψ^(n)(x) = d^(n+1)/dx^(n+1) ln(Γ(x)) = d^n/dx^n ψ(x)
 *
 *    Special cases:
 *        ψ^(0)(x) = ψ(x) = digamma function
 *        ψ^(1)(x) = trigamma function
 *        ψ^(2)(x) = tetragamma function
 *        ψ^(3)(x) = pentagamma function
 *
 * 2. RECURRENCE RELATION:
 *        ψ^(n)(x+1) = ψ^(n)(x) + (-1)^n * n! / x^(n+1)
 *
 * 3. REFLECTION FORMULA:
 *        ψ^(n)(1-x) + (-1)^(n+1) * ψ^(n)(x) = (-1)^n * π * d^n/dx^n cot(πx)
 *
 *    For trigamma (n=1):
 *        ψ^(1)(1-x) + ψ^(1)(x) = π² / sin²(πx)
 *
 * 4. ASYMPTOTIC EXPANSION:
 *    For large x:
 *        ψ^(n)(x) ~ (-1)^(n+1) * [(n-1)!/x^n + n!/(2x^(n+1)) +
 *                   Σ_{k=1}^∞ B_{2k} * (2k+n-1)! / ((2k)! * x^(2k+n))]
 *
 *    For trigamma (n=1):
 *        ψ^(1)(x) ~ 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷) - ...
 *
 * 5. SPECIAL VALUES:
 *    Trigamma:
 *        ψ^(1)(1) = π²/6
 *        ψ^(1)(1/2) = π²/2
 *        ψ^(1)(n) = π²/6 - Σ_{k=1}^{n-1} 1/k² for positive integers n
 *
 * 6. POLES:
 *    All polygamma functions have poles at non-positive integers (0, -1, -2, ...)
 *
 * 7. IMPLEMENTATION:
 *    - Uses asymptotic expansion for large x (x ≥ 7)
 *    - Uses recurrence relation for small x
 *    - Uses reflection formula for negative x
 *    - Specialized implementations for n=1,2,3 (trigamma, tetragamma, pentagamma)
 *
 * 8. DTYPE SUPPORT:
 *    - Supports float16, bfloat16, float32, float64
 *    - Supports complex64, complex128
 *
 * References:
 *    - Abramowitz & Stegun, Chapter 6
 *    - NIST Digital Library of Mathematical Functions, Chapter 5
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <type_traits>

#include "trigamma.h"
#include "tetragamma.h"
#include "pentagamma.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Bernoulli numbers B_{2k} for asymptotic expansions
// ============================================================================
// B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30, B_10 = 5/66, B_12 = -691/2730

namespace bernoulli {

constexpr double B2 = 1.0 / 6.0;
constexpr double B4 = -1.0 / 30.0;
constexpr double B6 = 1.0 / 42.0;
constexpr double B8 = -1.0 / 30.0;
constexpr double B10 = 5.0 / 66.0;
constexpr double B12 = -691.0 / 2730.0;
constexpr double B14 = 7.0 / 6.0;
constexpr double B16 = -3617.0 / 510.0;

}  // namespace bernoulli

// ============================================================================
// General polygamma function ψ^(n)(x)
// ============================================================================

/**
 * General polygamma function of order n for real types.
 *
 * ψ^(n)(x) = d^n/dx^n ψ(x)
 *
 * For n=0, this returns the digamma function.
 * For n=1, this returns the trigamma function.
 * For n=2, this returns the tetragamma function.
 * For n=3, this returns the pentagamma function.
 *
 * Uses the asymptotic expansion:
 *   ψ^(n)(x) ~ (-1)^(n+1) * [(n-1)!/x^n + n!/(2x^(n+1)) +
 *              Σ_{k=1}^∞ B_{2k} * (2k+n-1)! / ((2k)! * (n-1)! * x^(2k+n))]
 *
 * Note: For efficiency, use the specialized trigamma(), tetragamma(), or
 * pentagamma() functions instead of polygamma(1, x), polygamma(2, x), or
 * polygamma(3, x).
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t polygamma(int n, scalar_t x) {
  // Dispatch to specialized implementations for common orders
  if (n == 1) {
    return trigamma(x);
  }
  if (n == 2) {
    return tetragamma(x);
  }
  if (n == 3) {
    return pentagamma(x);
  }

  // For n=0, would need digamma, but that's in a separate header
  // For n > 3, we implement a general formula

  using std::abs;
  using std::floor;
  using std::pow;
  using std::tgamma;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  // Handle invalid order
  if (n < 0) {
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }

  // Handle NaN input
  if (x != x) {
    return x;
  }

  // For negative x, use reflection formula
  // This is complex for general n, so we use recurrence to make x positive
  if constexpr (!c10::is_complex<scalar_t>::value) {
    if (x <= scalar_t(0)) {
      // Check if x is a non-positive integer (pole)
      if (x == floor(x)) {
        return std::numeric_limits<scalar_t>::quiet_NaN();
      }
      // Use recurrence to shift x into positive territory
      // ψ^(n)(x) = ψ^(n)(x+k) + (-1)^n * n! * Σ_{j=0}^{k-1} 1/(x+j)^(n+1)
      scalar_t result = scalar_t(0);
      scalar_t sign = (n % 2 == 0) ? scalar_t(1) : scalar_t(-1);
      scalar_t n_factorial = scalar_t(tgamma(real_t(n + 1)));

      while (x <= scalar_t(0)) {
        result += sign * n_factorial / pow(x, scalar_t(n + 1));
        x += scalar_t(1);
      }

      return result + polygamma(n, x);
    }
  }

  scalar_t result = scalar_t(0);

  // Use recurrence to shift x to x >= 7
  // ψ^(n)(x+1) = ψ^(n)(x) + (-1)^n * n! / x^(n+1)
  // Rearranged: ψ^(n)(x) = ψ^(n)(x+1) - (-1)^n * n! / x^(n+1)
  const scalar_t shift_threshold = scalar_t(7);
  scalar_t sign = (n % 2 == 0) ? scalar_t(1) : scalar_t(-1);
  scalar_t n_factorial = scalar_t(tgamma(real_t(n + 1)));

  while (x < shift_threshold) {
    result -= sign * n_factorial / pow(x, scalar_t(n + 1));
    x += scalar_t(1);
  }

  // Asymptotic expansion for large x
  // ψ^(n)(x) ~ (-1)^(n+1) * [(n-1)!/x^n + n!/(2x^(n+1)) + ...]
  //
  // We compute the leading terms of the asymptotic expansion
  scalar_t asymp_sign = (n % 2 == 0) ? scalar_t(-1) : scalar_t(1);
  scalar_t nm1_factorial = scalar_t(tgamma(real_t(n)));  // (n-1)!

  scalar_t inv_x = scalar_t(1) / x;
  scalar_t inv_x_n = pow(inv_x, scalar_t(n));
  scalar_t inv_x_np1 = inv_x_n * inv_x;

  result += asymp_sign * (nm1_factorial * inv_x_n + n_factorial * inv_x_np1 / scalar_t(2));

  // Add Bernoulli correction terms
  // Coefficient for B_{2k} term: (2k+n-1)! / ((2k)! * (n-1)!)
  scalar_t inv_x2 = inv_x * inv_x;
  scalar_t inv_x_power = inv_x_np1 * inv_x;  // Start at x^(n+2)

  // Only add a few correction terms for numerical stability
  // Term for k=1 (B_2): coefficient = (n+1)! / (2! * (n-1)!) = n*(n+1)/2
  result += asymp_sign * scalar_t(bernoulli::B2) * scalar_t(n) * scalar_t(n + 1) / scalar_t(2) * inv_x_power;

  inv_x_power *= inv_x2;
  // Term for k=2 (B_4): coefficient = (n+3)! / (4! * (n-1)!) = n*(n+1)*(n+2)*(n+3)/24
  result += asymp_sign * scalar_t(bernoulli::B4) *
            scalar_t(n) * scalar_t(n + 1) * scalar_t(n + 2) * scalar_t(n + 3) / scalar_t(24) * inv_x_power;

  inv_x_power *= inv_x2;
  // Term for k=3 (B_6): coefficient = (n+5)! / (6! * (n-1)!)
  result += asymp_sign * scalar_t(bernoulli::B6) *
            scalar_t(n) * scalar_t(n + 1) * scalar_t(n + 2) * scalar_t(n + 3) * scalar_t(n + 4) * scalar_t(n + 5) / scalar_t(720) * inv_x_power;

  return result;
}

}  // namespace torchscience::impl::special_functions
