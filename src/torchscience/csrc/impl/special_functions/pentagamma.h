#pragma once

/*
 * Pentagamma Function ψ'''(x) = d⁴/dx⁴ ln(Γ(x))
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    The pentagamma function is the third derivative of the digamma function:
 *        ψ'''(x) = d³/dx³ ψ(x) = d⁴/dx⁴ ln(Γ(x))
 *
 * 2. RECURRENCE RELATION:
 *    ψ'''(x+1) = ψ'''(x) - 6/x⁴
 *
 * 3. REFLECTION FORMULA:
 *    ψ'''(1-x) + ψ'''(x) = -π d³/dx³ cot(πx)
 *                       = 2π⁴ (1 + 2cos²(πx)) / sin⁴(πx)
 *
 * 4. ASYMPTOTIC EXPANSION:
 *    For large x:
 *        ψ'''(x) ~ -2/x³ - 3/x⁴ - 2/x⁵ + Σ_{k=1}^∞ (2k+2)(2k+1) B_{2k} / x^(2k+3)
 *
 *    where B_{2k} are Bernoulli numbers.
 *
 * 5. SPECIAL VALUES:
 *    - ψ'''(1) = 6ζ(4) = π⁴/15 ≈ 6.494
 *    - ψ'''(x) has poles at x = 0, -1, -2, -3, ...
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

namespace pentagamma_detail {

constexpr double B2 = 1.0 / 6.0;
constexpr double B4 = -1.0 / 30.0;
constexpr double B6 = 1.0 / 42.0;
constexpr double B8 = -1.0 / 30.0;
constexpr double B10 = 5.0 / 66.0;

}  // namespace pentagamma_detail

// ============================================================================
// Pentagamma function ψ'''(x) = d⁴/dx⁴ ln(Γ(x))
// ============================================================================

/**
 * Pentagamma function for real types using asymptotic expansion and recurrence.
 *
 * Uses the asymptotic expansion for large x:
 *   ψ'''(x) ~ -2/x³ - 3/x⁴ - 2/x⁵ + Σ_{k=1}^∞ (2k+2)(2k+1) B_{2k} / x^(2k+3)
 *
 * The coefficients are (2k+2)(2k+1) * B_{2k}:
 *   k=1: 4*3 * B_2 = 12 * (1/6) = 2
 *   k=2: 6*5 * B_4 = 30 * (-1/30) = -1
 *   k=3: 8*7 * B_6 = 56 * (1/42) = 4/3
 *   k=4: 10*9 * B_8 = 90 * (-1/30) = -3
 *   k=5: 12*11 * B_10 = 132 * (5/66) = 10
 *
 * For small x, uses recurrence: ψ'''(x+1) = ψ'''(x) - 6/x⁴
 *   Rearranged: ψ'''(x) = ψ'''(x+1) + 6/x⁴
 *
 * For negative x, uses reflection formula:
 *   ψ'''(1-x) + ψ'''(x) = 2π⁴ (1 + 2cos²(πx)) / sin⁴(πx)
 *   So: ψ'''(x) = 2π⁴ (1 + 2cos²(πx)) / sin⁴(πx) - ψ'''(1-x)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t pentagamma(scalar_t x) {
  using std::abs;
  using std::floor;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t pi = real_t(3.14159265358979323846);
  const real_t pi4 = pi * pi * pi * pi;

  // Handle NaN
  if (x != x) {
    return x;
  }

  scalar_t result = scalar_t(0);

  // For negative x, use reflection formula
  // ψ'''(1-x) + ψ'''(x) = 2π⁴ (1 + 2cos²(πx)) / sin⁴(πx)
  // So: ψ'''(x) = 2π⁴ (1 + 2cos²(πx)) / sin⁴(πx) - ψ'''(1-x)
  if constexpr (!c10::is_complex<scalar_t>::value) {
    if (x <= scalar_t(0)) {
      // Check if x is a non-positive integer (pole)
      if (x == floor(x)) {
        return std::numeric_limits<scalar_t>::quiet_NaN();
      }
      return scalar_t(2) * pi4 * (scalar_t(1) + scalar_t(2) * (cos_pi(x) * cos_pi(x))) / (sin_pi(x) * sin_pi(x) * (sin_pi(x) * sin_pi(x))) - pentagamma(scalar_t(1) - x);
    }
  }

  // Use recurrence to shift x to x >= 7 for asymptotic expansion
  // Recurrence: ψ'''(x+1) = ψ'''(x) - 6/x⁴
  // Rearranged: ψ'''(x) = ψ'''(x+1) + 6/x⁴
  const scalar_t shift_threshold = scalar_t(7);
  while (x < shift_threshold) {
    scalar_t x2 = x * x;
    result += scalar_t(6) / (x2 * x2);
    x += scalar_t(1);
  }

  result += -scalar_t(2) * (scalar_t(1) / x * (scalar_t(1) / x) * (scalar_t(1) / x)) - scalar_t(3) * (scalar_t(1) / x * (scalar_t(1) / x)) * (scalar_t(1) / x * (scalar_t(1) / x)) - scalar_t(2) * (scalar_t(1) / x * (scalar_t(1) / x)) * (scalar_t(1) / x * (scalar_t(1) / x) * (scalar_t(1) / x));

  scalar_t inv_x5 = scalar_t(1) / x * (scalar_t(1) / x) * (scalar_t(1) / x * (scalar_t(1) / x) * (scalar_t(1) / x));
  result += inv_x5 * (
    scalar_t(12.0 * pentagamma_detail::B2) +                 // 12 * (1/6) = 2 / x⁵
    scalar_t(1) / x * (scalar_t(1) / x) * (
      scalar_t(30.0 * pentagamma_detail::B4) +               // 30 * (-1/30) = -1 / x⁷
      scalar_t(1) / x * (scalar_t(1) / x) * (
        scalar_t(56.0 * pentagamma_detail::B6) +             // 56 * (1/42) = 4/3 / x⁹
        scalar_t(1) / x * (scalar_t(1) / x) * (
          scalar_t(90.0 * pentagamma_detail::B8) +           // 90 * (-1/30) = -3 / x¹¹
          scalar_t(1) / x * (scalar_t(1) / x) * scalar_t(132.0 * pentagamma_detail::B10)  // 132 * (5/66) = 10 / x¹³
        )
      )
    )
  );

  return result;
}

/**
 * Complex pentagamma function.
 *
 * The pentagamma function has poles at non-positive integers (0, -1, -2, ...).
 * For complex numbers at these poles, we return NaN.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE c10::complex<T> pentagamma(c10::complex<T> x) {
  const T pi = T(3.14159265358979323846);
  const T pi4 = pi * pi * pi * pi;

  // Check for poles at non-positive integers (z = 0, -1, -2, ...)
  if (is_nonpositive_integer(x)) {
    return c10::complex<T>(
      std::numeric_limits<T>::quiet_NaN(),
      std::numeric_limits<T>::quiet_NaN()
    );
  }

  c10::complex<T> output(0, 0);

  if (x.real() < T(0.5)) {
    return c10::complex<T>(T(2) * pi4, 0) * (c10::complex<T>(1, 0) + c10::complex<T>(2, 0) * (cos_pi(x) * cos_pi(x))) / (sin_pi(x) * sin_pi(x) * (sin_pi(x) * sin_pi(x))) - pentagamma(c10::complex<T>(1, 0) - x);
  }

  // Use recurrence to shift x to Re(x) >= 7
  while (x.real() < T(7)) {
    auto x2 = x * x;
    output = output + c10::complex<T>(6, 0) / (x2 * x2);
    x = x + c10::complex<T>(1, 0);
  }

  return output - c10::complex<T>(2, 0) * (c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (c10::complex<T>(1, 0) / x)) - c10::complex<T>(3, 0) * (c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x)) * (c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x)) - c10::complex<T>(2, 0) * (c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x)) * (c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (c10::complex<T>(1, 0) / x)) + c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * (c10::complex<T>(1, 0) / x)) * ( c10::complex<T>(T(12.0 * pentagamma_detail::B2), 0) + c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * ( c10::complex<T>(T(30.0 * pentagamma_detail::B4), 0) + c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * ( c10::complex<T>(T(56.0 * pentagamma_detail::B6), 0) + c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * ( c10::complex<T>(T(90.0 * pentagamma_detail::B8), 0) + c10::complex<T>(1, 0) / x * (c10::complex<T>(1, 0) / x) * c10::complex<T>(T(132.0 * pentagamma_detail::B10), 0) ) ) ) );
}

}  // namespace torchscience::impl::special_functions
