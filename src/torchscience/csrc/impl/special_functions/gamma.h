#pragma once

/*
 * Gamma Function Γ(z)
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    For Re(z) > 0:
 *        Γ(z) = ∫₀^∞ t^(z-1) * e^(-t) dt
 *
 *    For other z (except non-positive integers):
 *        Analytic continuation via reflection formula:
 *        Γ(z) * Γ(1-z) = π / sin(πz)
 *
 * 2. SPECIAL VALUES:
 *    - Γ(n) = (n-1)! for positive integers n
 *    - Γ(1/2) = √π
 *    - Γ(z) has poles at z = 0, -1, -2, -3, ...
 *
 * 3. IMPLEMENTATION:
 *    - Uses lookup tables (LUTs) for positive integer arguments
 *      - Float32: Γ(1) to Γ(35) via 0! to 34! table
 *      - Float64: Γ(1) to Γ(171) via 0! to 170! table
 *    - Uses Lanczos approximation (g=7, n=9) for non-integer arguments
 *    - Loop-based Lanczos series computation for maintainability
 *    - Uses sin_pi() helper with range reduction for numerical stability
 *      at large negative arguments (e.g., z = -10^15)
 *    - Provides consistent results across CPU and CUDA
 *    - Half-precision types compute in float32 for accuracy
 *
 * 4. DERIVATIVE FORMULAS:
 *    First derivative:
 *        d/dz Γ(z) = Γ(z) * ψ(z)
 *    where ψ(z) is the digamma function.
 *
 *    Second derivative:
 *        d²/dz² Γ(z) = Γ(z) * (ψ(z)² + ψ'(z))
 *    where ψ'(z) is the trigamma function.
 *
 * 5. DTYPE SUPPORT:
 *    - Supports float16, bfloat16, float32, float64
 *    - Supports complex64, complex128
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>

#include "factorial.h"
#include "is_nonpositive_integer.h"
#include "lanczos_approximation.h"
#include "sin_pi.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Gamma function forward implementation
// ============================================================================

/**
 * Gamma function for float and double using LUT and Lanczos approximation.
 *
 * For positive integers, uses lookup table: Γ(n) = (n-1)!
 *   - Float: exact values for n = 1 to 35
 *   - Double: exact values for n = 1 to 171
 *
 * For non-integers, uses the Lanczos approximation with g=7:
 *   Γ(z+1) = √(2π) * (z + g + 0.5)^(z + 0.5) * e^(-(z + g + 0.5)) * A_g(z)
 *
 * where A_g(z) is a series approximation computed by lanczos_series().
 *
 * For negative z, uses the reflection formula with sin_pi() for numerical
 * stability at large negative values.
 *
 * For very large negative z where Γ(1-z) overflows to infinity,
 * returns zero (the mathematically correct limiting value).
 *
 * This implementation is device-portable (works on both CPU and CUDA).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<T>::value && (std::is_same_v<T, float> || std::is_same_v<T, double>), T> gamma(
  T z
) {
  using std::exp;
  using std::log;
  using std::floor;
  using std::isinf;

  const T pi = T(kPi);

  // Handle NaN
  if (z != z) {
    return z;
  }

  // Handle poles at non-positive integers
  if (z <= T(0) && z == floor(z)) {
    return std::numeric_limits<T>::infinity();
  }

  // Fast path for positive integers: Γ(n) = (n-1)!
  // Use LUT for exact results and better performance
  if (z > T(0) && z == floor(z)) {
    int n = static_cast<int>(z);
    if constexpr (std::is_same_v<T, double>) {
      if (n <= kGammaMaxIntDouble) {
        return T(kFactorialTableDouble[n - 1]);
      }
    } else {
      if (n <= kGammaMaxIntFloat) {
        return T(kFactorialTableFloat[n - 1]);
      }
    }
    // Beyond LUT range: overflow to infinity
    return std::numeric_limits<T>::infinity();
  }

  // Reflection formula for z < 0.5
  // Γ(z) = π / (sin(πz) * Γ(1-z))
  // Uses sin_pi() for numerical stability with large negative arguments
  if (z < T(0.5)) {
    T sin_pi_z = sin_pi(z);
    // Handle sin(pi*z) = 0 case (shouldn't happen after pole check, but be safe)
    if (sin_pi_z == T(0)) {
      return std::numeric_limits<T>::infinity();
    }

    T gamma_1_minus_z = gamma(T(1) - z);

    // Handle overflow case: when Γ(1-z) overflows to infinity,
    // Γ(z) approaches zero. This happens for very large negative z.
    // Mathematically: Γ(-n-α) = π / (sin(π(-n-α)) * Γ(n+1+α)) → 0 as n → ∞
    if (isinf(gamma_1_minus_z)) {
      return T(0);
    }

    return pi / (sin_pi_z * gamma_1_minus_z);
  }

  return T(kSqrt2Pi) * exp((z - T(0.5)) * log(z + T(kLanczosG) - T(0.5)) - (z + T(kLanczosG) - T(0.5))) * lanczos_series(z);
}

/**
 * Gamma function for half-precision types.
 * Computes in float32 for accuracy, then converts back.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t< !c10::is_complex<T>::value && !std::is_same_v<T, float> && !std::is_same_v<T, double>, T> gamma(
  T z
) {
  return static_cast<T>(gamma(static_cast<float>(z)));
}

// Forward declaration for complex gamma.
// This is required because the complex gamma implementation uses the reflection
// formula Γ(z) = π / (sin(πz) * Γ(1-z)) for Re(z) < 0.5, which recursively
// calls gamma() with a transformed argument. Without this forward declaration,
// the compiler cannot resolve the recursive call within the template.
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::enable_if_t<c10::is_complex<T>::value, T> gamma(
  T z
);

/**
 * Gamma function for complex types using Lanczos approximation.
 *
 * Uses the same Lanczos approximation as real types for consistency.
 * Uses sin_pi() for numerical stability with large negative real parts.
 * Uses lanczos_series() for maintainable loop-based computation.
 * Returns infinity at poles (z = 0, -1, -2, ...), consistent with real gamma.
 *
 * Special handling for:
 * - Very large negative Re(z): returns zero (the mathematically correct limit)
 * - z on real axis: uses real gamma to avoid inf*0=nan issues in complex exp
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::enable_if_t<c10::is_complex<T>::value, T> gamma(
  T z
) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::isinf;
  using std::isnan;

  using scalar_t = typename T::value_type;
  const scalar_t pi = scalar_t(kPi);

  // Handle NaN propagation: if either component is NaN, return NaN
  if (isnan(z.real()) || isnan(z.imag())) {
    return T(
      std::numeric_limits<scalar_t>::quiet_NaN(),
      std::numeric_limits<scalar_t>::quiet_NaN()
    );
  }

  // For z on or very close to the real axis, use real-valued computation
  // to avoid inf*0=nan issues that occur with complex exp when the result
  // overflows. This handles cases like gamma(1001.5 + 0j) correctly.
  const scalar_t imag_tolerance = std::numeric_limits<scalar_t>::epsilon() * scalar_t(100);
  if (abs(z.imag()) <= imag_tolerance) {
    scalar_t real_result = gamma(z.real());
    return T(real_result, scalar_t(0));
  }

  // Check for poles at non-positive integers (z = 0, -1, -2, ...)
  // Must check before reflection formula to avoid division by zero in sin(πz)
  if (is_nonpositive_integer(z)) {
    return T(
      std::numeric_limits<scalar_t>::infinity(),
      scalar_t(0)
    );
  }

  const auto real = [](scalar_t val) { return T(val, scalar_t(0)); };

  if (z.real() < scalar_t(0.5)) {
    auto gamma_1_minus_z = gamma(T(1, 0) - z);

    if (isinf(gamma_1_minus_z.real()) || isinf(gamma_1_minus_z.imag()) || isnan(gamma_1_minus_z.real()) || isnan(gamma_1_minus_z.imag())) {
      return T(scalar_t(0), scalar_t(0));
    }

    return real(pi) / (sin_pi(z) * gamma_1_minus_z);
  }

  return real(scalar_t(kSqrt2Pi)) * exp((z - real(scalar_t(0.5))) * log(z + real(scalar_t(kLanczosG) - scalar_t(0.5))) - (z + real(scalar_t(kLanczosG) - scalar_t(0.5)))) * lanczos_series(z);
}

}  // namespace torchscience::impl::special_functions
