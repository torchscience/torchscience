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
#include <tuple>

#include "digamma.h"
#include "trigamma.h"
#include "factorial.h"
#include "lanczos_approximation.h"

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

  using T = typename T::value_type;
  const T pi = T(kPi);

  // Handle NaN propagation: if either component is NaN, return NaN
  if (isnan(z.real()) || isnan(z.imag())) {
    return T(
      std::numeric_limits<T>::quiet_NaN(),
      std::numeric_limits<T>::quiet_NaN()
    );
  }

  // For z on or very close to the real axis, use real-valued computation
  // to avoid inf*0=nan issues that occur with complex exp when the result
  // overflows. This handles cases like gamma(1001.5 + 0j) correctly.
  const T imag_tolerance = std::numeric_limits<T>::epsilon() * T(100);
  if (abs(z.imag()) <= imag_tolerance) {
    T real_result = gamma(z.real());
    return T(real_result, T(0));
  }

  // Check for poles at non-positive integers (z = 0, -1, -2, ...)
  // Must check before reflection formula to avoid division by zero in sin(πz)
  if (is_nonpositive_integer(z)) {
    return T(
      std::numeric_limits<T>::infinity(),
      T(0)
    );
  }

  const auto real = [](T val) { return T(val, T(0)); };

  if (z.real() < T(0.5)) {
    auto gamma_1_minus_z = gamma(T(1, 0) - z);

    if (isinf(gamma_1_minus_z.real()) || isinf(gamma_1_minus_z.imag()) || isnan(gamma_1_minus_z.real()) || isnan(gamma_1_minus_z.imag())) {
      return T(T(0), T(0));
    }

    return real(pi) / (sin_pi(z) * gamma_1_minus_z);
  }

  return real(T(kSqrt2Pi)) * exp((z - real(T(0.5))) * log(z + real(T(kLanczosG) - T(0.5))) - (z + real(T(kLanczosG) - T(0.5)))) * lanczos_series(z);
}

// ============================================================================
// Backward implementation (first-order derivative)
// ============================================================================

/**
 * Backward pass for gamma function.
 * d/dz Γ(z) = Γ(z) * ψ(z)
 *
 * Returns gradient with respect to z.
 *
 * Complex Gradient Convention (Wirtinger Calculus):
 * ------------------------------------------------
 * PyTorch stores ∂L/∂z̄ (conjugate Wirtinger derivative) in .grad, not ∂L/∂z.
 * This convention makes gradient descent work correctly for complex parameters,
 * since the steepest descent direction for a real loss L is 2·∂L/∂z̄.
 *
 * For a holomorphic function f(z), the chain rule in Wirtinger calculus is:
 *     ∂L/∂z̄ = (∂L/∂w̄) · conj(df/dz)
 * where w = f(z).
 *
 * For Γ(z), the holomorphic derivative is dΓ/dz = Γ(z)·ψ(z), so:
 *     gradient_z̄ = gradient_w̄ · conj(Γ(z)·ψ(z))
 *
 * For real types, conjugation is the identity, so no special handling needed.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T gamma_backward(
  T gradient_output,
  T z
) {
  if constexpr (c10::is_complex<T>::value) {
    return gradient_output * std::conj(gamma(z) * digamma(z));
  }

  return gradient_output * gamma(z) * digamma(z);
}

// ============================================================================
// Double-backward implementation (second-order derivative)
// ============================================================================

/**
 * Double-backward pass for gamma function.
 *
 * Given:
 *   gradient_gradient_z = gradient w.r.t. gradient_z from first backward
 *   gradient_output = original upstream gradient
 *   z = original input
 *
 * The first backward computes:
 *   gradient_z = gradient_output * Γ(z) * ψ(z)
 *
 * Double backward computes:
 *   gradient_gradient_output = gradient_gradient_z * Γ(z) * ψ(z)  (derivative w.r.t gradient_output)
 *   gradient_z = gradient_gradient_z * gradient_output * d/dz[Γ(z) * ψ(z)]
 *              = gradient_gradient_z * gradient_output * Γ(z) * (ψ(z)² + ψ'(z))
 *
 * Returns: (gradient_gradient_output, gradient_z)
 *
 * Complex Gradient Convention (Wirtinger Calculus):
 * ------------------------------------------------
 * Same convention as gamma_backward applies here. For each term that involves
 * a holomorphic derivative, we conjugate when computing the Wirtinger gradient.
 *
 * gradient_gradient_output: This is ∂L/∂(gradient_output)̄, and since the first backward
 *   computed gradient_z = gradient_output * [Γ(z)·ψ(z)], differentiating w.r.t.
 *   gradient_output (treating it as a variable) gives conj(Γ(z)·ψ(z)).
 *
 * gradient_z: This differentiates through the [Γ(z)·ψ(z)] term, giving
 *   conj(d/dz[Γ(z)·ψ(z)]) = conj(Γ(z)·(ψ(z)² + ψ'(z))).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T> gamma_backward_backward(
  T gradient_gradient_z,
  T gradient_output,
  T z,
  const bool has_gradient_gradient_z
) {
  T gradient_gradient_output;
  T gradient_z;

  if (!has_gradient_gradient_z) {
    return std::make_tuple(T(0), T(0));
  }

  if constexpr (c10::is_complex<T>::value) {
    gradient_gradient_output = gradient_gradient_z * std::conj(gamma(z) * digamma(z));

    gradient_z = gradient_gradient_z * gradient_output * std::conj(gamma(z) * (digamma(z) * digamma(z) + trigamma(z)));
  } else {
    gradient_gradient_output = gradient_gradient_z * gamma(z) * digamma(z);

    gradient_z = gradient_gradient_z * gradient_output * (gamma(z) * (digamma(z) * digamma(z) + trigamma(z)));
  }

  return std::make_tuple(
    gradient_gradient_output,
    gradient_z
  );
}

}  // namespace torchscience::impl::special_functions
