#pragma once

/*
 * Non-Positive Integer Detection for Pole Handling
 *
 * DESIGN NOTES:
 *
 * 1. PURPOSE:
 *    Detects if a complex number is at or near a non-positive integer
 *    (0, -1, -2, -3, ...), which are poles of gamma-related functions.
 *
 * 2. TOLERANCE STRATEGY:
 *    Uses a hybrid absolute/relative tolerance approach:
 *    - For small values (|n| <= 1): uses absolute tolerance based on dtype precision
 *    - For larger values (|n| > 1): uses relative tolerance to account for
 *      floating-point representation error which grows with magnitude
 *
 * 3. USE CASES:
 *    - Gamma function Γ(x)
 *    - Digamma function ψ(x)
 *    - Trigamma function ψ'(x)
 *    - Tetragamma function ψ''(x)
 *    - Pentagamma function ψ'''(x)
 *    - All higher-order polygamma functions
 *
 * 4. DTYPE SUPPORT:
 *    - Supports float and double precision
 *    - Works with c10::complex<float> and c10::complex<double>
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <type_traits>

namespace torchscience::impl::special_functions {

// ============================================================================
// Tolerance constants for pole detection
// ============================================================================

// Tolerance for detecting non-positive integer poles in double precision.
// Double epsilon ≈ 2.2e-16; we use ~10000x epsilon for robustness against
// accumulated floating-point errors in the reflection and recurrence formulas.
constexpr double kPoleDetectionToleranceDouble = 1e-12;

// Tolerance for detecting non-positive integer poles in single precision.
// Float epsilon ≈ 1.2e-7; we use ~100x epsilon for robustness.
// Also used for half/bfloat16 which compute in float internally.
constexpr float kPoleDetectionToleranceFloat = 1e-5f;

// ============================================================================
// Non-positive integer detection
// ============================================================================

/**
 * Check if a complex number is at a non-positive integer (pole of gamma/digamma/trigamma).
 *
 * Returns true if z is approximately equal to 0, -1, -2, -3, ...
 *
 * The detection uses a hybrid absolute/relative tolerance approach:
 * - For small values (|n| <= 1): uses absolute tolerance based on dtype precision
 * - For larger values (|n| > 1): uses relative tolerance to account for
 *   floating-point representation error which grows with magnitude
 *
 * This ensures correct pole detection even for large negative integers like -1000,
 * where floating-point representation error can exceed a fixed absolute tolerance.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE bool is_nonpositive_integer(c10::complex<T> z) {
  using std::abs;
  using std::trunc;

  // Type-appropriate base tolerance using named constants
  T base_tol;
  if constexpr (std::is_same_v<T, double>) {
    base_tol = T(kPoleDetectionToleranceDouble);
  } else {
    base_tol = T(kPoleDetectionToleranceFloat);
  }

  // Check if imaginary part is approximately zero (absolute tolerance is fine here)
  if (abs(z.imag()) > base_tol) {
    return false;
  }

  T real_part = z.real();

  // Use trunc to find the integer part, avoiding floor(x + 0.5) ambiguity
  // at half-integers. trunc rounds toward zero, which is correct for
  // checking if we're near an integer.
  T truncated = trunc(real_part);

  // Check if we're closer to truncated or truncated ± 1
  T candidates[3] = {truncated - T(1), truncated, truncated + T(1)};
  T min_dist = abs(real_part - candidates[0]);
  T nearest_int = candidates[0];

  for (int i = 1; i < 3; ++i) {
    T dist = abs(real_part - candidates[i]);
    if (dist < min_dist) {
      min_dist = dist;
      nearest_int = candidates[i];
    }
  }

  // Use relative tolerance for larger magnitudes, absolute for small
  // This handles floating-point representation error which scales with magnitude
  T tol = base_tol * (T(1) + abs(nearest_int));

  if (min_dist > tol) {
    return false;
  }

  // Check if that integer is non-positive (0, -1, -2, ...)
  return nearest_int <= T(0);
}

}  // namespace torchscience::impl::special_functions
