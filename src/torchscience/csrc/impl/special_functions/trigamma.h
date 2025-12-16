#pragma once

/*
 * Trigamma Function ψ'(x) = d²/dx² ln(Γ(x))
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    The trigamma function is the derivative of the digamma function:
 *        ψ'(x) = d/dx ψ(x) = d²/dx² ln(Γ(x))
 *
 * 2. SPECIAL VALUES:
 *    - ψ'(1) = π²/6
 *    - ψ'(n) = π²/6 - Σ_{k=1}^{n-1} 1/k² for positive integers n
 *    - ψ'(x) has poles at x = 0, -1, -2, -3, ...
 *
 * 3. IMPLEMENTATION:
 *    - Uses asymptotic expansion for large x (x ≥ 6)
 *    - Uses recurrence relation for small x: ψ'(x+1) = ψ'(x) - 1/x²
 *    - Uses reflection formula for negative x: ψ'(1-x) + ψ'(x) = π²/sin²(πx)
 *
 * 4. DTYPE SUPPORT:
 *    - Supports float16, bfloat16, float32, float64
 *    - Supports complex64, complex128
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <type_traits>

#include "digamma.h"  // for is_nonpositive_integer

namespace torchscience::impl::special_functions {

// ============================================================================
// Trigamma function ψ'(x) = d²/dx² ln(Γ(x))
// ============================================================================

/**
 * Trigamma function for real types using asymptotic expansion and recurrence.
 *
 * Uses the asymptotic expansion for large x:
 *   ψ'(x) ≈ 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷) - ...
 *
 * For small x, uses recurrence: ψ'(x+1) = ψ'(x) - 1/x²
 * For negative x, uses reflection: ψ'(1-x) + ψ'(x) = π²/sin²(πx)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t trigamma(scalar_t x) {
  using std::floor;
  using std::sin;
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t pi = real_t(3.14159265358979323846);

  // Handle special cases
  if (x != x) {  // NaN check
    return x;
  }

  scalar_t result = scalar_t(0);

  // For negative x, use reflection formula
  if constexpr (!c10::is_complex<scalar_t>::value) {
    if (x <= scalar_t(0)) {
      // Check if x is a non-positive integer (pole)
      if (x == floor(x)) {
        return std::numeric_limits<scalar_t>::quiet_NaN();
      }
      // Reflection: ψ'(1-x) + ψ'(x) = π²/sin²(πx)
      // So: ψ'(x) = π²/sin²(πx) - ψ'(1-x)
      // Use range-reduced sin_pi for numerical stability with large negative x
      scalar_t sin_pi_x = sin_pi(x);
      scalar_t pi_csc_sq = (pi * pi) / (sin_pi_x * sin_pi_x);
      return pi_csc_sq - trigamma(scalar_t(1) - x);
    }
  }

  // Use recurrence to shift x to x >= 6 for asymptotic expansion
  const scalar_t shift_threshold = scalar_t(6);
  while (x < shift_threshold) {
    result += scalar_t(1) / (x * x);
    x += scalar_t(1);
  }

  // Asymptotic expansion for large x
  //
  // ψ'(x) ≈ 1/x + 1/(2x²) + Σ_{k=1}^∞ B_{2k}/x^{2k+1}
  //
  // where B_{2k} are Bernoulli numbers: B_2=1/6, B_4=-1/30, B_6=1/42, B_8=-1/30, B_10=5/66
  //
  // The coefficients used are |B_{2k}|:
  //   k=1: |B_2|  = 1/6
  //   k=2: |B_4|  = 1/30
  //   k=3: |B_6|  = 1/42
  //   k=4: |B_8|  = 1/30
  //   k=5: |B_10| = 5/66
  //
  // The alternating signs of Bernoulli numbers (+, -, +, -, +) are handled by
  // the nested subtraction structure combined with the leading plus sign.
  scalar_t inv_x = scalar_t(1) / x;
  scalar_t inv_x2 = inv_x * inv_x;

  result += inv_x + inv_x2 / scalar_t(2);
  result += inv_x2 * inv_x * (
    scalar_t(1.0 / 6.0) -                // |B_2| = 1/6
    inv_x2 * (
      scalar_t(1.0 / 30.0) -             // |B_4| = 1/30
      inv_x2 * (
        scalar_t(1.0 / 42.0) -           // |B_6| = 1/42
        inv_x2 * (
          scalar_t(1.0 / 30.0) -         // |B_8| = 1/30
          inv_x2 * scalar_t(5.0 / 66.0)  // |B_10| = 5/66
        )
      )
    )
  );

  return result;
}

/**
 * Complex trigamma function.
 *
 * The trigamma function has poles at non-positive integers (0, -1, -2, ...).
 * For complex numbers at these poles, we return NaN.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE c10::complex<T> trigamma(c10::complex<T> x) {
  using std::sin;

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
  // Use range-reduced sin_pi for numerical stability with large negative Re(x)
  if (x.real() < T(0.5)) {
    auto sin_pi_x = sin_pi(x);
    auto pi_csc_sq = c10::complex<T>(pi * pi, 0) / (sin_pi_x * sin_pi_x);
    return pi_csc_sq - trigamma(c10::complex<T>(1, 0) - x);
  }

  // Use recurrence to shift x to Re(x) >= 6
  while (x.real() < T(6)) {
    result = result + c10::complex<T>(1, 0) / (x * x);
    x = x + c10::complex<T>(1, 0);
  }

  // Asymptotic expansion
  auto inv_x = c10::complex<T>(1, 0) / x;
  auto inv_x2 = inv_x * inv_x;

  result = result + inv_x + inv_x2 / c10::complex<T>(2, 0);
  result = result + inv_x2 * inv_x * (
    c10::complex<T>(T(1.0 / 6.0), 0) -
    inv_x2 * (
      c10::complex<T>(T(1.0 / 30.0), 0) -
      inv_x2 * (
        c10::complex<T>(T(1.0 / 42.0), 0) -
        inv_x2 * (
          c10::complex<T>(T(1.0 / 30.0), 0) -
          inv_x2 * c10::complex<T>(T(5.0 / 66.0), 0)
        )
      )
    )
  );

  return result;
}

}  // namespace torchscience::impl::special_functions
