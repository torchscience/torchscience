#pragma once

/*
 * Numerically Stable cos(pi*x) Computation
 *
 * DESIGN NOTES:
 *
 * 1. PURPOSE:
 *    Compute cos(pi*x) with high accuracy, especially for large |x| where
 *    direct computation loses precision due to floating-point limitations.
 *
 * 2. RANGE REDUCTION:
 *    For large |x|, uses remainder(x, 2) to reduce to [-1, 1] before
 *    computing cos(pi*r). This works because cos(pi*x) has period 2.
 *
 * 3. SPECIAL CASES:
 *    - cos_pi(n) = (-1)^n for any integer n
 *    - cos_pi(n + 0.5) = 0 for any integer n
 *    - cos_pi(NaN) = NaN
 *    - cos_pi(+/-Inf) = NaN
 *
 * 4. DTYPE SUPPORT:
 *    - float, double (native precision)
 *    - float16, bfloat16 (computed in float32)
 *    - complex64, complex128
 */

#include "sin_pi.h"

namespace torchscience::impl::special_functions {

/**
 * Compute cos(pi*x) with range reduction for numerical stability.
 *
 * Uses the same range reduction approach as sin_pi for large arguments.
 *
 * Special cases:
 *   - cos_pi(n) = (-1)^n for any integer n
 *   - cos_pi(n + 0.5) = 0 for any integer n
 *   - cos_pi(NaN) = NaN
 *   - cos_pi(+/-Inf) = NaN
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<scalar_t>::value &&
  (std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>),
  scalar_t>
cos_pi(scalar_t x) {
  using std::remainder;
  using std::cos;
  using std::floor;
  using std::abs;
  using std::fmod;

  const scalar_t pi = scalar_t(kPi);

  // Handle NaN
  if (x != x) {
    return x;
  }

  // Handle infinity -> NaN (cos of infinity is undefined)
  if (x == std::numeric_limits<scalar_t>::infinity() ||
      x == -std::numeric_limits<scalar_t>::infinity()) {
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }

  // For small |x|, direct computation is accurate enough
  if (abs(x) < scalar_t(1e8)) {
    // Handle exact integers -> exact +/-1
    if (x == floor(x)) {
      // cos(pi*n) = (-1)^n
      // Use fmod to get n mod 2 safely for large integers
      scalar_t n_mod_2 = fmod(abs(x), scalar_t(2));
      return (n_mod_2 < scalar_t(0.5)) ? scalar_t(1) : scalar_t(-1);
    }
    // Handle exact half-integers -> exact 0
    scalar_t x_plus_half = x + scalar_t(0.5);
    if (x_plus_half == floor(x_plus_half)) {
      return scalar_t(0);
    }
    return cos(pi * x);
  }

  // Range reduction for large |x|
  // remainder(x, 2) gives r in [-1, 1] such that x = 2k + r for some integer k
  // Since cos(pi*x) has period 2: cos(pi*x) = cos(pi*r)
  scalar_t r = remainder(x, scalar_t(2));

  // Handle the case where r is an integer
  if (r == floor(r)) {
    scalar_t n_mod_2 = fmod(abs(r), scalar_t(2));
    return (n_mod_2 < scalar_t(0.5)) ? scalar_t(1) : scalar_t(-1);
  }

  // Handle exact half-integers
  scalar_t r_plus_half = r + scalar_t(0.5);
  if (r_plus_half == floor(r_plus_half)) {
    return scalar_t(0);
  }

  return cos(pi * r);
}

/**
 * Compute cos(pi*x) for half-precision types.
 * Computes in float32 for accuracy, then converts back.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<scalar_t>::value &&
  !std::is_same_v<scalar_t, float> &&
  !std::is_same_v<scalar_t, double>,
  scalar_t>
cos_pi(scalar_t x) {
  // Compute in float32 for better accuracy
  return static_cast<scalar_t>(cos_pi(static_cast<float>(x)));
}

/**
 * Compute cos(pi*z) for complex z with numerical stability.
 *
 * Uses the identity: cos(pi*(a+bi)) = cos(pi*a)cosh(pi*b) - i*sin(pi*a)sinh(pi*b)
 *
 * For large |Re(z)|, we use range reduction on the real part to maintain
 * accuracy, similar to the real-valued cos_pi.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> cos_pi(c10::complex<T> z) {
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

  // cos(pi(a+bi)) = cos(pi*a)cosh(pi*b) - i*sin(pi*a)sinh(pi*b)
  T sin_pi_a = sin(pi * reduced_a);
  T cos_pi_a = cos(pi * reduced_a);
  T sinh_pi_b = sinh(pi * b);
  T cosh_pi_b = cosh(pi * b);

  return c10::complex<T>(
    cos_pi_a * cosh_pi_b,
    -sin_pi_a * sinh_pi_b
  );
}

}  // namespace torchscience::impl::special_functions
