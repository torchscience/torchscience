#pragma once

/*
 * Tetragamma Function ψ''(x) = d³/dx³ ln(Γ(x))
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    The tetragamma function is the second derivative of the digamma function:
 *        ψ''(x) = d²/dx² ψ(x) = d³/dx³ ln(Γ(x))
 *
 * 2. RECURRENCE RELATION:
 *    ψ''(x+1) = ψ''(x) + 2/x³
 *
 * 3. REFLECTION FORMULA:
 *    ψ''(1-x) - ψ''(x) = 2π³ cos(πx) / sin³(πx)
 *
 * 4. ASYMPTOTIC EXPANSION:
 *    For large x:
 *        ψ''(x) ~ -1/x² - 1/x³ - 1/(2x⁴) + Σ_{k=1}^∞ (2k+1) B_{2k} / x^(2k+2)
 *
 *    where B_{2k} are Bernoulli numbers.
 *
 * 5. SPECIAL VALUES:
 *    - ψ''(1) = -2ζ(3) ≈ -2.404
 *    - ψ''(x) has poles at x = 0, -1, -2, -3, ...
 *
 * 6. IMPLEMENTATION:
 *    - Uses asymptotic expansion for large x (x ≥ 7)
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

namespace torchscience::impl::special_functions {

// ============================================================================
// Bernoulli numbers B_{2k} for asymptotic expansions
// ============================================================================

namespace tetragamma_detail {

constexpr double B2 = 1.0 / 6.0;
constexpr double B4 = -1.0 / 30.0;
constexpr double B6 = 1.0 / 42.0;
constexpr double B8 = -1.0 / 30.0;
constexpr double B10 = 5.0 / 66.0;

}  // namespace tetragamma_detail

// ============================================================================
// Tetragamma function ψ''(x) = d³/dx³ ln(Γ(x))
// ============================================================================

/**
 * Tetragamma function for real types using asymptotic expansion and recurrence.
 *
 * Uses the asymptotic expansion for large x:
 *   ψ''(x) ~ -1/x² - 1/x³ - 1/(2x⁴) + Σ_{k=1}^∞ (2k+1) B_{2k} / x^(2k+2)
 *
 * The coefficients are (2k+1) * B_{2k}:
 *   k=1: 3 * B_2 = 3 * (1/6) = 1/2
 *   k=2: 5 * B_4 = 5 * (-1/30) = -1/6
 *   k=3: 7 * B_6 = 7 * (1/42) = 1/6
 *   k=4: 9 * B_8 = 9 * (-1/30) = -3/10
 *   k=5: 11 * B_10 = 11 * (5/66) = 5/6
 *
 * For small x, uses recurrence: ψ''(x+1) = ψ''(x) + 2/x³
 *   Rearranged: ψ''(x) = ψ''(x+1) - 2/x³
 *
 * For negative x, uses reflection formula:
 *   ψ''(1-x) - ψ''(x) = 2π³ cos(πx) / sin³(πx)
 *   So: ψ''(x) = ψ''(1-x) - 2π³ cos(πx) / sin³(πx)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t tetragamma(scalar_t x) {
  using std::abs;
  using std::floor;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t pi = real_t(3.14159265358979323846);

  // Handle NaN
  if (x != x) {
    return x;
  }

  scalar_t result = scalar_t(0);

  // For negative x, use reflection formula
  // ψ''(1-x) - ψ''(x) = 2π³ cos(πx) / sin³(πx)
  // So: ψ''(x) = ψ''(1-x) - 2π³ cos(πx) / sin³(πx)
  if constexpr (!c10::is_complex<scalar_t>::value) {
    if (x <= scalar_t(0)) {
      // Check if x is a non-positive integer (pole)
      if (x == floor(x)) {
        return std::numeric_limits<scalar_t>::quiet_NaN();
      }
      // Use range-reduced sin_pi and cos_pi for numerical stability
      scalar_t sin_pi_x = sin_pi(x);
      scalar_t cos_pi_x = cos_pi(x);
      scalar_t sin_cubed = sin_pi_x * sin_pi_x * sin_pi_x;
      scalar_t reflection_term = scalar_t(2) * pi * pi * pi * cos_pi_x / sin_cubed;
      return tetragamma(scalar_t(1) - x) - reflection_term;
    }
  }

  // Use recurrence to shift x to x >= 7 for asymptotic expansion
  // Recurrence: ψ''(x+1) = ψ''(x) + 2/x³
  // Rearranged: ψ''(x) = ψ''(x+1) - 2/x³
  const scalar_t shift_threshold = scalar_t(7);
  while (x < shift_threshold) {
    result -= scalar_t(2) / (x * x * x);
    x += scalar_t(1);
  }

  // Asymptotic expansion for large x
  //
  // ψ''(x) ~ -1/x² - 1/x³ - 1/(2x⁴) + Σ_{k=1}^∞ (2k+1) B_{2k} / x^(2k+2)
  //

  // Start with -1/x² - 1/x³ - 1/(2x⁴)
  result += -(scalar_t(1) / x * (scalar_t(1) / x)) - scalar_t(1) / x * (scalar_t(1) / x) * (scalar_t(1) / x) - scalar_t(1) / x * (scalar_t(1) / x) * (scalar_t(1) / x * (scalar_t(1) / x)) / scalar_t(2);

  // Add Bernoulli terms starting at 1/x⁴
  scalar_t inv_x4 = scalar_t(1) / x * (scalar_t(1) / x) * (scalar_t(1) / x * (scalar_t(1) / x));
  result += inv_x4 * (
    scalar_t(3.0 * tetragamma_detail::B2) +               // 3 * (1/6) = 0.5 / x⁴
    scalar_t(1) / x * (scalar_t(1) / x) * (
      scalar_t(5.0 * tetragamma_detail::B4) +             // 5 * (-1/30) = -1/6 / x⁶
      scalar_t(1) / x * (scalar_t(1) / x) * (
        scalar_t(7.0 * tetragamma_detail::B6) +           // 7 * (1/42) = 1/6 / x⁸
        scalar_t(1) / x * (scalar_t(1) / x) * (
          scalar_t(9.0 * tetragamma_detail::B8) +         // 9 * (-1/30) = -3/10 / x¹⁰
          scalar_t(1) / x * (scalar_t(1) / x) * scalar_t(11.0 * tetragamma_detail::B10) // 11 * (5/66) = 5/6 / x¹²
        )
      )
    )
  );

  return result;
}

/**
 * Complex tetragamma function.
 *
 * The tetragamma function has poles at non-positive integers (0, -1, -2, ...).
 * For complex numbers at these poles, we return NaN.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE c10::complex<T> tetragamma(c10::complex<T> x) {
  const T pi = T(3.14159265358979323846);

  // Check for poles at non-positive integers (z = 0, -1, -2, ...)
  if (is_nonpositive_integer(x)) {
    return c10::complex<T>(
      std::numeric_limits<T>::quiet_NaN(),
      std::numeric_limits<T>::quiet_NaN()
    );
  }

  c10::complex<T> result(0, 0);

  if (x.real() < T(0.5)) {
    return tetragamma(c10::complex<T>(1, 0) - x) - c10::complex<T>(T(2) * pi * pi * pi, 0) * cos_pi(x) / (sin_pi(x) * sin_pi(x) * sin_pi(x));
  }

  // Use recurrence to shift x to Re(x) >= 7
  while (x.real() < T(7)) {
    result = result - c10::complex<T>(2, 0) / (x * x * x);
    x = x + c10::complex<T>(1, 0);
  }

  result = result - c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) - c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (c10::complex<T>(1, 0) / x) - c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x)) / c10::complex<T>(2, 0);

  return result + c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x)) * (
           c10::complex<T>(T(3.0 * tetragamma_detail::B2), 0) +
           c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (
             c10::complex<T>(T(5.0 * tetragamma_detail::B4), 0) +
             c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (
               c10::complex<T>(T(7.0 * tetragamma_detail::B6), 0) +
               c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (
                 c10::complex<T>(T(9.0 * tetragamma_detail::B8), 0) +
                 c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * c10::complex<T>(T(11.0 * tetragamma_detail::B10), 0)
               )
             )
           )
         );
}

}  // namespace torchscience::impl::special_functions
