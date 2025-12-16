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
#include <type_traits>

#include "digamma.h"
#include "trigamma.h"
#include "factorial.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Lanczos coefficients (g=7, n=9)
// ============================================================================

// Lanczos approximation parameter
constexpr double kLanczosG = 7.0;

// Number of Lanczos coefficients
constexpr int kLanczosN = 9;

// Lanczos coefficients for g=7, n=9
// These provide ~15 digits of precision for double
constexpr double kLanczosCoeffs[kLanczosN] = {
  0.99999999999980993227684700473478,
  676.520368121885098567009190444019,
  -1259.13921672240287047156078755283,
  771.3234287776530788486528258894,
  -176.61502916214059906584551354,
  12.507343278686904814458936853,
  -0.13857109526572011689554707,
  9.984369578019570859563e-6,
  1.50563273514931155834e-7
};

// sqrt(2 * pi)
constexpr double kSqrt2Pi = 2.5066282746310005024157652848110452530069867406099;

// Note: kPi, sin_pi, and cos_pi are defined in digamma.h
// Note: Factorial LUTs (kFactorialTableFloat/Double, kGammaMaxIntFloat/Double)
//       are defined in factorial.h

// ============================================================================
// Lanczos series computation
// ============================================================================

/**
 * Compute the Lanczos series A_g(z) for real types.
 *
 * A_g(z) = c_0 + Σ_{i=1}^{n-1} c_i / (z + i - 1)
 *
 * where z is the original argument (not z-1).
 *
 * For the Lanczos formula Γ(z+1) = √(2π) * t^(z+0.5) * e^(-t) * A_g(z),
 * the denominators in A_g are: z, z+1, z+2, ..., z+n-2
 * which corresponds to: (z-1)+1, (z-1)+2, ..., (z-1)+(n-1)
 *
 * This loop-based implementation improves maintainability over the
 * unrolled version while maintaining the same numerical accuracy.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<scalar_t>::value &&
  (std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>),
  scalar_t>
lanczos_series(scalar_t z) {
  // Start with c_0
  scalar_t A_g = scalar_t(kLanczosCoeffs[0]);

  // Add terms c_i / (z + i - 1) for i = 1 to n-1
  // Denominators are: z, z+1, z+2, ..., z+n-2
  for (int i = 1; i < kLanczosN; ++i) {
    A_g += scalar_t(kLanczosCoeffs[i]) / (z + scalar_t(i - 1));
  }

  return A_g;
}

/**
 * Compute the Lanczos series A_g(z) for complex types.
 *
 * Same formula as the real version but for complex arguments.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> lanczos_series(c10::complex<T> z) {
  // Helper to create real-valued complex constants
  const auto real = [](T val) { return c10::complex<T>(val, T(0)); };

  // Start with c_0
  c10::complex<T> A_g = real(T(kLanczosCoeffs[0]));

  // Add terms c_i / (z + i - 1) for i = 1 to n-1
  // Denominators are: z, z+1, z+2, ..., z+n-2
  for (int i = 1; i < kLanczosN; ++i) {
    A_g = A_g + real(T(kLanczosCoeffs[i])) / (z + real(T(i - 1)));
  }

  return A_g;
}

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
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<scalar_t>::value &&
  (std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>),
  scalar_t>
gamma(scalar_t z) {
  using std::exp;
  using std::log;
  using std::floor;
  using std::isinf;

  const scalar_t pi = scalar_t(kPi);

  // Handle NaN
  if (z != z) {
    return z;
  }

  // Handle poles at non-positive integers
  if (z <= scalar_t(0) && z == floor(z)) {
    return std::numeric_limits<scalar_t>::infinity();
  }

  // Fast path for positive integers: Γ(n) = (n-1)!
  // Use LUT for exact results and better performance
  if (z > scalar_t(0) && z == floor(z)) {
    int n = static_cast<int>(z);
    if constexpr (std::is_same_v<scalar_t, double>) {
      if (n <= kGammaMaxIntDouble) {
        return scalar_t(kFactorialTableDouble[n - 1]);
      }
    } else {
      if (n <= kGammaMaxIntFloat) {
        return scalar_t(kFactorialTableFloat[n - 1]);
      }
    }
    // Beyond LUT range: overflow to infinity
    return std::numeric_limits<scalar_t>::infinity();
  }

  // Reflection formula for z < 0.5
  // Γ(z) = π / (sin(πz) * Γ(1-z))
  // Uses sin_pi() for numerical stability with large negative arguments
  if (z < scalar_t(0.5)) {
    scalar_t sin_pi_z = sin_pi(z);
    // Handle sin(pi*z) = 0 case (shouldn't happen after pole check, but be safe)
    if (sin_pi_z == scalar_t(0)) {
      return std::numeric_limits<scalar_t>::infinity();
    }

    scalar_t gamma_1_minus_z = gamma(scalar_t(1) - z);

    // Handle overflow case: when Γ(1-z) overflows to infinity,
    // Γ(z) approaches zero. This happens for very large negative z.
    // Mathematically: Γ(-n-α) = π / (sin(π(-n-α)) * Γ(n+1+α)) → 0 as n → ∞
    if (isinf(gamma_1_minus_z)) {
      return scalar_t(0);
    }

    return pi / (sin_pi_z * gamma_1_minus_z);
  }

  // Lanczos approximation for non-integer z >= 0.5
  const scalar_t g = scalar_t(kLanczosG);
  const scalar_t sqrt_2pi = scalar_t(kSqrt2Pi);

  // Compute the Lanczos series using the loop-based helper
  scalar_t A_g = lanczos_series(z);

  // Compute t = z - 1 + g + 0.5 = z + g - 0.5
  scalar_t t = z + g - scalar_t(0.5);

  // Γ(z) = √(2π) * t^(z - 0.5) * e^(-t) * A_g(z)
  // Use log form for numerical stability: exp((z - 0.5) * log(t) - t)
  scalar_t log_t = log(t);
  scalar_t result = sqrt_2pi * exp((z - scalar_t(0.5)) * log_t - t) * A_g;

  return result;
}

/**
 * Gamma function for half-precision types.
 * Computes in float32 for accuracy, then converts back.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<scalar_t>::value &&
  !std::is_same_v<scalar_t, float> &&
  !std::is_same_v<scalar_t, double>,
  scalar_t>
gamma(scalar_t z) {
  // Compute in float32 for better accuracy
  return static_cast<scalar_t>(gamma(static_cast<float>(z)));
}

// Forward declaration for complex gamma.
// This is required because the complex gamma implementation uses the reflection
// formula Γ(z) = π / (sin(πz) * Γ(1-z)) for Re(z) < 0.5, which recursively
// calls gamma() with a transformed argument. Without this forward declaration,
// the compiler cannot resolve the recursive call within the template.
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<c10::is_complex<scalar_t>::value, scalar_t>
gamma(scalar_t z);

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
template <typename scalar_t> C10_HOST_DEVICE C10_ALWAYS_INLINE std::enable_if_t<c10::is_complex<scalar_t>::value, scalar_t> gamma(scalar_t z) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::isinf;
  using std::isnan;

  using T = typename scalar_t::value_type;
  const T pi = T(kPi);

  // Handle NaN propagation: if either component is NaN, return NaN
  if (isnan(z.real()) || isnan(z.imag())) {
    return scalar_t(
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
    return scalar_t(real_result, T(0));
  }

  // Check for poles at non-positive integers (z = 0, -1, -2, ...)
  // Must check before reflection formula to avoid division by zero in sin(πz)
  if (is_nonpositive_integer(z)) {
    return scalar_t(
      std::numeric_limits<T>::infinity(),
      T(0)
    );
  }

  // Helper to create real-valued complex constants
  const auto real = [](T val) { return scalar_t(val, T(0)); };

  // Reflection formula for Re(z) < 0.5
  // Uses sin_pi() for numerical stability with large negative real parts
  if (z.real() < T(0.5)) {
    // Γ(z) = π / (sin(πz) * Γ(1-z))
    auto sin_pi_z = sin_pi(z);
    auto gamma_1_minus_z = gamma(scalar_t(1, 0) - z);

    // Handle overflow/nan case: when Γ(1-z) is infinite or NaN,
    // Γ(z) approaches zero. This happens for very large negative Re(z).
    // Mathematically: Γ(-n-α) = π / (sin(π(-n-α)) * Γ(n+1+α)) → 0 as n → ∞
    if (isinf(gamma_1_minus_z.real()) || isinf(gamma_1_minus_z.imag()) ||
        isnan(gamma_1_minus_z.real()) || isnan(gamma_1_minus_z.imag())) {
      // Return zero - the mathematically correct limit
      return scalar_t(T(0), T(0));
    }

    return real(pi) / (sin_pi_z * gamma_1_minus_z);
  }

  // Lanczos approximation for Re(z) >= 0.5
  const T g = T(kLanczosG);
  const T sqrt_2pi = T(kSqrt2Pi);

  // Compute the Lanczos series using the loop-based helper
  scalar_t A_g = lanczos_series(z);

  // Compute t = z + g - 0.5
  const auto t = z + real(g - T(0.5));

  // Γ(z) = √(2π) * t^(z - 0.5) * e^(-t) * A_g(z)
  // Use log form for numerical stability
  const auto result = real(sqrt_2pi) *
                      exp((z - real(T(0.5))) * log(t) - t) * A_g;

  return result;
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
 *     grad_z̄ = grad_w̄ · conj(Γ(z)·ψ(z))
 *
 * For real types, conjugation is the identity, so no special handling needed.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
gamma_backward(scalar_t grad_output, scalar_t z) {
  auto gamma_z = gamma(z);
  auto psi_z = digamma(z);

  if constexpr (c10::is_complex<scalar_t>::value) {
    // Wirtinger chain rule: conjugate the holomorphic derivative
    return grad_output * std::conj(gamma_z * psi_z);
  } else {
    return grad_output * gamma_z * psi_z;
  }
}

// ============================================================================
// Double-backward implementation (second-order derivative)
// ============================================================================

/**
 * Double-backward pass for gamma function.
 *
 * Given:
 *   gg_z = gradient w.r.t. gradient_z from first backward
 *   grad_output = original upstream gradient
 *   z = original input
 *
 * The first backward computes:
 *   gradient_z = grad_output * Γ(z) * ψ(z)
 *
 * Double backward computes:
 *   gradient_grad_output = gg_z * Γ(z) * ψ(z)  (derivative w.r.t grad_output)
 *   gradient_z = gg_z * grad_output * d/dz[Γ(z) * ψ(z)]
 *              = gg_z * grad_output * Γ(z) * (ψ(z)² + ψ'(z))
 *
 * Returns: (gradient_grad_output, gradient_z)
 *
 * Complex Gradient Convention (Wirtinger Calculus):
 * ------------------------------------------------
 * Same convention as gamma_backward applies here. For each term that involves
 * a holomorphic derivative, we conjugate when computing the Wirtinger gradient.
 *
 * gradient_grad_output: This is ∂L/∂(grad_output)̄, and since the first backward
 *   computed gradient_z = grad_output * [Γ(z)·ψ(z)], differentiating w.r.t.
 *   grad_output (treating it as a variable) gives conj(Γ(z)·ψ(z)).
 *
 * gradient_z: This differentiates through the [Γ(z)·ψ(z)] term, giving
 *   conj(d/dz[Γ(z)·ψ(z)]) = conj(Γ(z)·(ψ(z)² + ψ'(z))).
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t>
gamma_backward_backward(
    scalar_t gg_z,
    scalar_t grad_output,
    scalar_t z,
    bool has_gg_z
) {
  scalar_t gradient_grad_output = scalar_t(0);
  scalar_t gradient_z = scalar_t(0);

  if (!has_gg_z) {
    return std::make_tuple(gradient_grad_output, gradient_z);
  }

  auto gamma_z = gamma(z);
  auto psi_z = digamma(z);
  auto psi_prime_z = trigamma(z);

  // d/dz[Γ(z) * ψ(z)] = Γ(z) * ψ(z) * ψ(z) + Γ(z) * ψ'(z)
  //                   = Γ(z) * (ψ(z)² + ψ'(z))
  auto dGamma_psi_dz = gamma_z * (psi_z * psi_z + psi_prime_z);

  if constexpr (c10::is_complex<scalar_t>::value) {
    // Wirtinger chain rule: conjugate each holomorphic derivative term
    gradient_grad_output = gg_z * std::conj(gamma_z * psi_z);
    gradient_z = gg_z * grad_output * std::conj(dGamma_psi_dz);
  } else {
    gradient_grad_output = gg_z * gamma_z * psi_z;
    gradient_z = gg_z * grad_output * dGamma_psi_dz;
  }

  return std::make_tuple(gradient_grad_output, gradient_z);
}

}  // namespace torchscience::impl::special_functions
