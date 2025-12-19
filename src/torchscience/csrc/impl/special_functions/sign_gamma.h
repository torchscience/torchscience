#pragma once

/*
 * Gamma Function Sign Computation
 *
 * DESIGN NOTES:
 *
 * 1. PURPOSE:
 *    Computes the sign of the Gamma function for real arguments.
 *    This is essential for computing Gamma ratios using lgamma,
 *    which returns log(|Gamma(x)|) and loses sign information.
 *
 * 2. MATHEMATICAL BACKGROUND:
 *    For real x:
 *    - Gamma(x) > 0 for x > 0
 *    - Gamma(x) alternates sign between consecutive negative integers:
 *      * Gamma(x) < 0 for x in (-1, 0)
 *      * Gamma(x) > 0 for x in (-2, -1)
 *      * Gamma(x) < 0 for x in (-3, -2)
 *      * etc.
 *    - Gamma(x) has poles at non-positive integers (0, -1, -2, ...)
 *
 * 3. FORMULA:
 *    For x < 0 (non-integer):
 *      sign(Gamma(x)) = (-1)^ceil(-x) = (-1)^(floor(-x) + 1)
 *
 *    Derivation using reflection formula:
 *      Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
 *      For x in (-1, 0): 1-x in (1, 2), so Gamma(1-x) > 0
 *                        sin(pi*x) < 0, so Gamma(x) < 0
 *
 * 4. USE CASES:
 *    - Computing Gamma ratios: Gamma(a)/Gamma(b) = sign * exp(lgamma(a) - lgamma(b))
 *    - Hypergeometric function transformations (DLMF 15.8.2, 15.8.4)
 *    - Beta function and related special functions
 */

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cmath>

namespace torchscience::impl::special_functions {

// Helper trait to check if type is a floating point type (including PyTorch types)
template <typename T>
struct is_floating_point_or_pytorch_float : std::is_floating_point<T> {};

template <>
struct is_floating_point_or_pytorch_float<c10::Half> : std::true_type {};

template <>
struct is_floating_point_or_pytorch_float<c10::BFloat16> : std::true_type {};

/**
 * Compute the sign of Gamma(x) for real x.
 *
 * Returns:
 *   +1 if Gamma(x) > 0
 *   -1 if Gamma(x) < 0
 *
 * Undefined behavior for x at non-positive integers (poles of Gamma).
 *
 * Examples:
 *   sign_gamma(2.5)  = +1  (Gamma(2.5) = 1.329...)
 *   sign_gamma(0.5)  = +1  (Gamma(0.5) = sqrt(pi))
 *   sign_gamma(-0.5) = -1  (Gamma(-0.5) = -3.544...)
 *   sign_gamma(-1.5) = +1  (Gamma(-1.5) = 2.363...)
 *   sign_gamma(-2.5) = -1  (Gamma(-2.5) = -0.945...)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE int sign_gamma(T x) {
  static_assert(is_floating_point_or_pytorch_float<T>::value, "sign_gamma requires floating point type");

  using std::floor;

  // Convert to float for Half/BFloat16 types to use standard math functions
  float x_f = static_cast<float>(x);

  if (x_f > 0.0f) {
    return 1;
  }

  // For x < 0 (non-integer):
  // sign(Gamma(x)) = (-1)^ceil(-x)
  //
  // Since ceil(-x) = floor(-x) + 1 for non-integer -x,
  // we compute k = floor(-x) + 1 and return (-1)^k
  int k = static_cast<int>(floor(-x_f)) + 1;
  return (k % 2 == 0) ? 1 : -1;
}

/**
 * Compute exp(lgamma(x)) with correct sign.
 *
 * This is equivalent to |Gamma(x)| * sign(Gamma(x)) = Gamma(x),
 * but computed via lgamma for numerical stability with large arguments.
 *
 * Returns Gamma(x) computed as sign_gamma(x) * exp(lgamma(x)).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T signed_exp_lgamma(T x) {
  using std::exp;
  using std::lgamma;

  return T(static_cast<float>(sign_gamma(x)) * exp(lgamma(static_cast<float>(x))));
}

/**
 * Compute a ratio of Gamma functions using lgamma with correct sign.
 *
 * Computes: Gamma(a) / Gamma(b) = sign_gamma(a) / sign_gamma(b) * exp(lgamma(a) - lgamma(b))
 *
 * This avoids overflow for large arguments where Gamma values would overflow.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T gamma_ratio(T a, T b) {
  using std::exp;
  using std::lgamma;

  return T(static_cast<float>(sign_gamma(a) * sign_gamma(b)) * exp(lgamma(static_cast<float>(a)) - lgamma(static_cast<float>(b))));
}

/**
 * Compute a ratio of Gamma function products using lgamma with correct sign.
 *
 * Computes: (Gamma(a1) * Gamma(a2)) / (Gamma(b1) * Gamma(b2))
 *
 * This is a common pattern in hypergeometric function transformations.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T gamma_ratio_4(T a1, T a2, T b1, T b2) {
  using std::exp;
  using std::lgamma;

  return T(static_cast<float>(sign_gamma(a1) * sign_gamma(a2) * sign_gamma(b1) * sign_gamma(b2)) * exp(lgamma(static_cast<float>(a1)) + lgamma(static_cast<float>(a2)) - lgamma(static_cast<float>(b1)) - lgamma(static_cast<float>(b2))));
}

}  // namespace torchscience::impl::special_functions
