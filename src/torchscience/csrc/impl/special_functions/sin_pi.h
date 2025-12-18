#pragma once

/*
 * Numerically Stable sin(pi*x) Computation
 *
 * DESIGN NOTES:
 *
 * 1. PURPOSE:
 *    Compute sin(pi*x) with high accuracy, especially for large |x| where
 *    direct computation loses precision due to floating-point limitations.
 *
 * 2. RANGE REDUCTION:
 *    For large |x|, uses remainder(x, 2) to reduce to [-1, 1] before
 *    computing sin(pi*r). This works because sin(pi*x) has period 2.
 *
 * 3. SPECIAL CASES:
 *    - sin_pi(n) = 0 for any integer n
 *    - sin_pi(n + 0.5) = +/-1 for any integer n
 *    - sin_pi(NaN) = NaN
 *    - sin_pi(+/-Inf) = NaN
 *
 * 4. DTYPE SUPPORT:
 *    - float, double (native precision)
 *    - float16, bfloat16 (computed in float32)
 *    - complex64, complex128
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <type_traits>

namespace torchscience::impl::special_functions {

// pi constant
constexpr double kPi = 3.14159265358979323846;

/**
 * Compute sin(pi*x) with range reduction for numerical stability.
 *
 * For large |x|, direct computation of sin(pi*x) loses precision because
 * floating-point representation cannot accurately capture the fractional
 * part. For example, at x = 10^15, double precision has ~0.22 ULP error,
 * meaning values differing by less than ~0.22 cannot be distinguished.
 *
 * This function uses range reduction to compute sin(pi*x) accurately:
 *   1. Reduce x to r in [-1, 1] using IEEE remainder: r = remainder(x, 2)
 *   2. Compute sin(pi*r) directly
 *
 * This works because sin(pi*x) has period 2, and remainder() is computed
 * with higher internal precision.
 *
 * Special cases:
 *   - sin_pi(n) = 0 for any integer n
 *   - sin_pi(n + 0.5) = +/-1 for any integer n
 *   - sin_pi(NaN) = NaN
 *   - sin_pi(+/-Inf) = NaN
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<scalar_t>::value &&
  (std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>),
  scalar_t>
sin_pi(scalar_t x) {
  using std::remainder;
  using std::sin;
  using std::floor;
  using std::abs;

  const scalar_t pi = scalar_t(kPi);

  // Handle NaN
  if (x != x) {
    return x;
  }

  // Handle infinity -> NaN (sin of infinity is undefined)
  if (x == std::numeric_limits<scalar_t>::infinity() ||
      x == -std::numeric_limits<scalar_t>::infinity()) {
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }

  // For small |x|, direct computation is accurate enough
  // Threshold chosen where range reduction overhead isn't worth it
  // and floating-point precision is sufficient
  if (abs(x) < scalar_t(1e8)) {
    // Handle exact integers -> exact zero
    if (x == floor(x)) {
      return scalar_t(0);
    }
    return sin(pi * x);
  }

  // Range reduction for large |x|
  // remainder(x, 2) gives r in [-1, 1] such that x = 2k + r for some integer k
  // Since sin(pi*x) has period 2: sin(pi*x) = sin(pi*r)
  scalar_t r = remainder(x, scalar_t(2));

  // Handle the case where r rounds to an integer (should give exact zero)
  if (r == floor(r)) {
    return scalar_t(0);
  }

  return sin(pi * r);
}

/**
 * Compute sin(pi*x) for half-precision types.
 * Computes in float32 for accuracy, then converts back.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<scalar_t>::value &&
  !std::is_same_v<scalar_t, float> &&
  !std::is_same_v<scalar_t, double>,
  scalar_t>
sin_pi(scalar_t x) {
  // Compute in float32 for better accuracy
  return static_cast<scalar_t>(sin_pi(static_cast<float>(x)));
}

/**
 * Compute sin(pi*z) for complex z with numerical stability.
 *
 * Uses the identity: sin(pi*(a+bi)) = sin(pi*a)cosh(pi*b) + i*cos(pi*a)sinh(pi*b)
 *
 * For large |Re(z)|, we use range reduction on the real part to maintain
 * accuracy, similar to the real-valued sin_pi.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> sin_pi(c10::complex<T> z) {
  using std::sin;
  using std::cos;
  using std::sinh;
  using std::cosh;
  using std::remainder;
  using std::abs;

  const T pi = T(kPi);
  T a = z.real();
  T b = z.imag();

  // Range reduce the real part for numerical stability
  T reduced_a;
  if (abs(a) < T(1e8)) {
    reduced_a = a;
  } else {
    reduced_a = remainder(a, T(2));
  }

  return c10::complex<T>(
    sin(pi * reduced_a) * cosh(pi * b),
    cos(pi * reduced_a) * sinh(pi * b)
  );
}

}  // namespace torchscience::impl::special_functions
