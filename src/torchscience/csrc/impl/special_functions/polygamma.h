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

#include "sin_pi.h"
#include "cos_pi.h"

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
// Trigamma function ψ^(1)(x) = d/dx ψ(x) = d²/dx² ln(Γ(x))
// ============================================================================

/**
 * Trigamma function for real types.
 *
 * Uses the asymptotic expansion for large x:
 *   ψ^(1)(x) ~ 1/x + 1/(2x²) + Σ_{k=1}^∞ B_{2k} / x^(2k+1)
 *
 * where B_{2k} are Bernoulli numbers.
 *
 * Expanding:
 *   ψ^(1)(x) ~ 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷) - 1/(30x⁹) + 5/(66x¹¹) - ...
 *
 * For small x, uses recurrence: ψ^(1)(x+1) = ψ^(1)(x) - 1/x²
 * For negative x, uses reflection: ψ^(1)(1-x) + ψ^(1)(x) = π² / sin²(πx)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t trigamma(scalar_t x) {
  using std::abs;
  using std::floor;
  using std::sin;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t pi = real_t(3.14159265358979323846);
  const real_t pi_squared = pi * pi;

  // Handle NaN
  if (x != x) {
    return x;
  }

  scalar_t result = scalar_t(0);

  // For negative x, use reflection formula: ψ^(1)(1-x) + ψ^(1)(x) = π² / sin²(πx)
  // So: ψ^(1)(x) = π² / sin²(πx) - ψ^(1)(1-x)
  if constexpr (!c10::is_complex<scalar_t>::value) {
    if (x <= scalar_t(0)) {
      // Check if x is a non-positive integer (pole)
      if (x == floor(x)) {
        return std::numeric_limits<scalar_t>::quiet_NaN();
      }
      // Reflection formula using range-reduced sin_pi for numerical stability
      scalar_t sin_pi_x = sin_pi(x);
      scalar_t csc_squared = scalar_t(1) / (sin_pi_x * sin_pi_x);
      return pi_squared * csc_squared - trigamma(scalar_t(1) - x);
    }
  }

  // Use recurrence to shift x to x >= 7 for asymptotic expansion
  // Recurrence: ψ^(1)(x+1) = ψ^(1)(x) - 1/x²
  // Rearranged: ψ^(1)(x) = ψ^(1)(x+1) + 1/x²
  const scalar_t shift_threshold = scalar_t(7);
  while (x < shift_threshold) {
    result += scalar_t(1) / (x * x);
    x += scalar_t(1);
  }

  // Asymptotic expansion for large x
  //
  // ψ^(1)(x) ~ 1/x + 1/(2x²) + Σ_{k=1}^∞ B_{2k} / x^(2k+1)
  //
  // = 1/x + 1/(2x²) + B_2/x³ + B_4/x⁵ + B_6/x⁷ + B_8/x⁹ + B_10/x¹¹ + ...
  // = 1/x + 1/(2x²) + (1/6)/x³ - (1/30)/x⁵ + (1/42)/x⁷ - (1/30)/x⁹ + (5/66)/x¹¹ - ...
  //
  scalar_t inv_x = scalar_t(1) / x;
  scalar_t inv_x2 = inv_x * inv_x;

  // Start with 1/x + 1/(2x²)
  result += inv_x + inv_x2 / scalar_t(2);

  // Add terms from Bernoulli numbers
  scalar_t inv_x3 = inv_x2 * inv_x;
  result += inv_x3 * (
    scalar_t(bernoulli::B2) +                    // 1/6 / x³
    inv_x2 * (
      scalar_t(bernoulli::B4) +                  // -1/30 / x⁵
      inv_x2 * (
        scalar_t(bernoulli::B6) +                // 1/42 / x⁷
        inv_x2 * (
          scalar_t(bernoulli::B8) +              // -1/30 / x⁹
          inv_x2 * (
            scalar_t(bernoulli::B10) +           // 5/66 / x¹¹
            inv_x2 * scalar_t(bernoulli::B12)    // -691/2730 / x¹³
          )
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
  const T pi_squared = pi * pi;

  // Type-appropriate tolerance for pole detection
  T tol;
  if constexpr (std::is_same_v<T, double>) {
    tol = T(1e-12);
  } else {
    tol = T(1e-5f);
  }

  // Check for poles at non-positive integers
  if (std::abs(x.imag()) < tol) {
    T real_part = x.real();
    T nearest_int = std::round(real_part);
    if (std::abs(real_part - nearest_int) < tol * (T(1) + std::abs(nearest_int))) {
      if (nearest_int <= T(0)) {
        return c10::complex<T>(
          std::numeric_limits<T>::quiet_NaN(),
          std::numeric_limits<T>::quiet_NaN()
        );
      }
    }
  }

  c10::complex<T> result(0, 0);

  // For Re(x) < 0.5, use reflection formula
  if (x.real() < T(0.5)) {
    // ψ^(1)(x) = π² / sin²(πx) - ψ^(1)(1-x)
    auto sin_pi_x = sin_pi(x);
    auto csc_squared = c10::complex<T>(1, 0) / (sin_pi_x * sin_pi_x);
    return c10::complex<T>(pi_squared, 0) * csc_squared - trigamma(c10::complex<T>(1, 0) - x);
  }

  // Use recurrence to shift x to Re(x) >= 7
  while (x.real() < T(7)) {
    result = result + c10::complex<T>(1, 0) / (x * x);
    x = x + c10::complex<T>(1, 0);
  }

  // Asymptotic expansion
  auto inv_x = c10::complex<T>(1, 0) / x;
  auto inv_x2 = inv_x * inv_x;

  result = result + inv_x + inv_x2 / c10::complex<T>(2, 0);

  auto inv_x3 = inv_x2 * inv_x;
  result = result + inv_x3 * (
    c10::complex<T>(T(bernoulli::B2), 0) +
    inv_x2 * (
      c10::complex<T>(T(bernoulli::B4), 0) +
      inv_x2 * (
        c10::complex<T>(T(bernoulli::B6), 0) +
        inv_x2 * (
          c10::complex<T>(T(bernoulli::B8), 0) +
          inv_x2 * (
            c10::complex<T>(T(bernoulli::B10), 0) +
            inv_x2 * c10::complex<T>(T(bernoulli::B12), 0)
          )
        )
      )
    )
  );

  return result;
}

// ============================================================================
// Tetragamma function ψ^(2)(x) = d²/dx² ψ(x) = d³/dx³ ln(Γ(x))
// ============================================================================

/**
 * Tetragamma function for real types.
 *
 * Uses the asymptotic expansion for large x:
 *   ψ^(2)(x) ~ -1/x² - 1/x³ - 1/(2x⁴) + Σ_{k=1}^∞ (2k+1) * B_{2k} / x^(2k+2)
 *
 * For small x, uses recurrence: ψ^(2)(x+1) = ψ^(2)(x) + 2/x³
 * For negative x, uses reflection formula.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t tetragamma(scalar_t x) {
  using std::abs;
  using std::floor;
  using std::sin;
  using std::cos;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t pi = real_t(3.14159265358979323846);

  // Handle NaN
  if (x != x) {
    return x;
  }

  scalar_t result = scalar_t(0);

  // For negative x, use reflection formula
  // ψ^(2)(1-x) - ψ^(2)(x) = 2π³ * cos(πx) / sin³(πx)
  // So: ψ^(2)(x) = ψ^(2)(1-x) - 2π³ * cos(πx) / sin³(πx)
  if constexpr (!c10::is_complex<scalar_t>::value) {
    if (x <= scalar_t(0)) {
      // Check if x is a non-positive integer (pole)
      if (x == floor(x)) {
        return std::numeric_limits<scalar_t>::quiet_NaN();
      }
      scalar_t sin_pi_x = sin_pi(x);
      scalar_t cos_pi_x = cos_pi(x);
      scalar_t sin_cubed = sin_pi_x * sin_pi_x * sin_pi_x;
      scalar_t reflection_term = scalar_t(2) * pi * pi * pi * cos_pi_x / sin_cubed;
      return tetragamma(scalar_t(1) - x) - reflection_term;
    }
  }

  // Use recurrence to shift x to x >= 7
  // Recurrence: ψ^(2)(x+1) = ψ^(2)(x) + 2/x³
  // Rearranged: ψ^(2)(x) = ψ^(2)(x+1) - 2/x³
  const scalar_t shift_threshold = scalar_t(7);
  while (x < shift_threshold) {
    result -= scalar_t(2) / (x * x * x);
    x += scalar_t(1);
  }

  // Asymptotic expansion for large x
  //
  // ψ^(2)(x) ~ -1/x² - 1/x³ - 1/(2x⁴) + Σ_{k=1}^∞ (2k+1) * B_{2k} / x^(2k+2)
  //
  // The coefficients are (2k+1) * B_{2k}:
  //   k=1: 3 * B_2 = 3 * (1/6) = 1/2
  //   k=2: 5 * B_4 = 5 * (-1/30) = -1/6
  //   k=3: 7 * B_6 = 7 * (1/42) = 1/6
  //   k=4: 9 * B_8 = 9 * (-1/30) = -3/10
  //   k=5: 11 * B_10 = 11 * (5/66) = 5/6
  //
  scalar_t inv_x = scalar_t(1) / x;
  scalar_t inv_x2 = inv_x * inv_x;

  // Start with -1/x² - 1/x³ - 1/(2x⁴)
  result += -inv_x2 - inv_x2 * inv_x - inv_x2 * inv_x2 / scalar_t(2);

  // Add Bernoulli terms starting at 1/x⁴
  scalar_t inv_x4 = inv_x2 * inv_x2;
  result += inv_x4 * (
    scalar_t(3.0 * bernoulli::B2) +               // 3 * (1/6) = 0.5 / x⁴
    inv_x2 * (
      scalar_t(5.0 * bernoulli::B4) +             // 5 * (-1/30) = -1/6 / x⁶
      inv_x2 * (
        scalar_t(7.0 * bernoulli::B6) +           // 7 * (1/42) = 1/6 / x⁸
        inv_x2 * (
          scalar_t(9.0 * bernoulli::B8) +         // 9 * (-1/30) = -3/10 / x¹⁰
          inv_x2 * scalar_t(11.0 * bernoulli::B10) // 11 * (5/66) = 5/6 / x¹²
        )
      )
    )
  );

  return result;
}

/**
 * Complex tetragamma function.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE c10::complex<T> tetragamma(c10::complex<T> x) {
  using std::sin;
  using std::cos;

  const T pi = T(3.14159265358979323846);

  // Type-appropriate tolerance for pole detection
  T tol;
  if constexpr (std::is_same_v<T, double>) {
    tol = T(1e-12);
  } else {
    tol = T(1e-5f);
  }

  // Check for poles at non-positive integers
  if (std::abs(x.imag()) < tol) {
    T real_part = x.real();
    T nearest_int = std::round(real_part);
    if (std::abs(real_part - nearest_int) < tol * (T(1) + std::abs(nearest_int))) {
      if (nearest_int <= T(0)) {
        return c10::complex<T>(
          std::numeric_limits<T>::quiet_NaN(),
          std::numeric_limits<T>::quiet_NaN()
        );
      }
    }
  }

  c10::complex<T> result(0, 0);

  // For Re(x) < 0.5, use reflection formula
  if (x.real() < T(0.5)) {
    auto sin_pi_x = sin_pi(x);
    auto cos_pi_x = cos_pi(x);
    auto sin_cubed = sin_pi_x * sin_pi_x * sin_pi_x;
    auto reflection_term = c10::complex<T>(T(2) * pi * pi * pi, 0) * cos_pi_x / sin_cubed;
    return tetragamma(c10::complex<T>(1, 0) - x) - reflection_term;
  }

  // Use recurrence to shift x to Re(x) >= 7
  while (x.real() < T(7)) {
    result = result - c10::complex<T>(2, 0) / (x * x * x);
    x = x + c10::complex<T>(1, 0);
  }

  // Asymptotic expansion
  auto inv_x = c10::complex<T>(1, 0) / x;
  auto inv_x2 = inv_x * inv_x;

  result = result - inv_x2 - inv_x2 * inv_x - inv_x2 * inv_x2 / c10::complex<T>(2, 0);

  auto inv_x4 = inv_x2 * inv_x2;
  result = result + inv_x4 * (
    c10::complex<T>(T(3.0 * bernoulli::B2), 0) +
    inv_x2 * (
      c10::complex<T>(T(5.0 * bernoulli::B4), 0) +
      inv_x2 * (
        c10::complex<T>(T(7.0 * bernoulli::B6), 0) +
        inv_x2 * (
          c10::complex<T>(T(9.0 * bernoulli::B8), 0) +
          inv_x2 * c10::complex<T>(T(11.0 * bernoulli::B10), 0)
        )
      )
    )
  );

  return result;
}

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
 *
 * Uses the asymptotic expansion:
 *   ψ^(n)(x) ~ (-1)^(n+1) * [(n-1)!/x^n + n!/(2x^(n+1)) +
 *              Σ_{k=1}^∞ B_{2k} * (2k+n-1)! / ((2k)! * (n-1)! * x^(2k+n))]
 *
 * Note: For efficiency, use the specialized trigamma() or tetragamma()
 * functions instead of polygamma(1, x) or polygamma(2, x).
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

  // For n=0, would need digamma, but that's in a separate header
  // For n > 2, we implement a general formula

  using std::abs;
  using std::floor;
  using std::pow;
  using std::tgamma;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t pi = real_t(3.14159265358979323846);

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
