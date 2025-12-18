#pragma once

/*
 * Digamma Function ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x)
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    The digamma function is the logarithmic derivative of the gamma function:
 *        ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x)
 *
 * 2. SPECIAL VALUES:
 *    - ψ(1) = -γ (negative Euler-Mascheroni constant ≈ -0.5772)
 *    - ψ(n) = -γ + Σ_{k=1}^{n-1} 1/k for positive integers n
 *    - ψ(x) has poles at x = 0, -1, -2, -3, ...
 *
 * 3. IMPLEMENTATION:
 *    - Uses asymptotic expansion for large x (x ≥ 6)
 *    - Uses recurrence relation for small x: ψ(x+1) = ψ(x) + 1/x
 *    - Uses reflection formula for negative x: ψ(1-x) - ψ(x) = π*cot(πx)
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
#include "cos_pi.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Digamma function ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x)
// ============================================================================

/**
 * Digamma function for real types using asymptotic expansion and recurrence.
 *
 * Uses the asymptotic expansion for large x:
 *   ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) + ...
 *
 * For small x, uses recurrence: ψ(x+1) = ψ(x) + 1/x
 * For negative x, uses reflection: ψ(1-x) = ψ(x) + π*cot(πx)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t digamma(scalar_t x) {
  using std::log;
  using std::floor;
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t pi = real_t(3.14159265358979323846);

  // Handle special cases
  if (x != x) {  // NaN check
    return x;
  }

  scalar_t result = scalar_t(0);

  // For negative x, use reflection formula: ψ(1-x) - ψ(x) = π*cot(πx)
  // So: ψ(x) = ψ(1-x) - π*cot(πx)
  if constexpr (!c10::is_complex<scalar_t>::value) {
    if (x <= scalar_t(0)) {
      // Check if x is a non-positive integer (pole)
      if (x == floor(x)) {
        return std::numeric_limits<scalar_t>::quiet_NaN();
      }
      // Reflection: ψ(x) = ψ(1-x) - π*cot(πx)
      // Use range-reduced sin_pi/cos_pi for numerical stability with large negative x
      scalar_t cot_pi_x = cos_pi(x) / sin_pi(x);
      return digamma(scalar_t(1) - x) - pi * cot_pi_x;
    }
  }

  // Use recurrence to shift x to x >= 6 for asymptotic expansion
  const scalar_t shift_threshold = scalar_t(6);
  while (x < shift_threshold) {
    result -= scalar_t(1) / x;
    x += scalar_t(1);
  }

  // Asymptotic expansion for large x
  //
  // ψ(x) ≈ ln(x) - 1/(2x) - Σ_{k=1}^∞ B_{2k}/(2k · x^{2k})
  //
  // where B_{2k} are Bernoulli numbers: B_2=1/6, B_4=-1/30, B_6=1/42, B_8=-1/30, B_10=5/66
  //
  // The coefficients used are |B_{2k}|/(2k):
  //   k=1: |B_2|/2  = (1/6)/2   = 1/12
  //   k=2: |B_4|/4  = (1/30)/4  = 1/120
  //   k=3: |B_6|/6  = (1/42)/6  = 1/252
  //   k=4: |B_8|/8  = (1/30)/8  = 1/240
  //   k=5: |B_10|/10 = (5/66)/10 = 5/660
  //
  // The alternating signs of Bernoulli numbers (-, +, -, +, -) are handled by
  // the nested subtraction structure combined with the leading minus sign.
  scalar_t inv_x = scalar_t(1) / x;
  scalar_t inv_x2 = inv_x * inv_x;

  result += log(x) - inv_x / scalar_t(2);
  result -= inv_x2 * (
    scalar_t(1.0 / 12.0) -               // |B_2|/2 = 1/12
    inv_x2 * (
      scalar_t(1.0 / 120.0) -            // |B_4|/4 = 1/120
      inv_x2 * (
        scalar_t(1.0 / 252.0) -          // |B_6|/6 = 1/252
        inv_x2 * (
          scalar_t(1.0 / 240.0) -        // |B_8|/8 = 1/240
          inv_x2 * scalar_t(5.0 / 660.0) // |B_10|/10 = 5/660
        )
      )
    )
  );

  return result;
}

/**
 * Complex digamma function using the same asymptotic expansion.
 *
 * The digamma function has poles at non-positive integers (0, -1, -2, ...).
 * For complex numbers at these poles, we return NaN.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE c10::complex<T> digamma(c10::complex<T> x) {
  using std::log;

  const T pi = T(3.14159265358979323846);

  // Check for poles at non-positive integers (z = 0, -1, -2, ...)
  if (is_nonpositive_integer(x)) {
    return c10::complex<T>(
      std::numeric_limits<T>::quiet_NaN(),
      std::numeric_limits<T>::quiet_NaN()
    );
  }

  c10::complex<T> result(0, 0);

  // For Re(x) < 0.5, use reflection formula
  if (x.real() < T(0.5)) {
    return digamma(c10::complex<T>(1, 0) - x) - c10::complex<T>(pi, 0) * (cos_pi(x) / sin_pi(x));
  }

  // Use recurrence to shift x to Re(x) >= 6
  while (x.real() < T(6)) {
    result -= c10::complex<T>(1, 0) / x;
    x += c10::complex<T>(1, 0);
  }

  result = result + log(x) - c10::complex<T>(1, 0) / x / c10::complex<T>(2, 0);

  return result - c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (
           c10::complex<T>(T(1.0 / 12.0), 0) -
           c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (
             c10::complex<T>(T(1.0 / 120.0), 0) -
             c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (
               c10::complex<T>(T(1.0 / 252.0), 0) -
               c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (
                 c10::complex<T>(T(1.0 / 240.0), 0) -
                 c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * c10::complex<T>(T(5.0 / 660.0), 0)
               )
             )
           )
         );
}

}  // namespace torchscience::impl::special_functions
