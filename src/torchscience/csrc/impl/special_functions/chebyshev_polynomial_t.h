#pragma once

/*
 * Chebyshev Polynomial of the First Kind T_v(z)
 *
 * DESIGN NOTES:
 *
 * 1. DISPATCH LOGIC:
 *    - Path A (Recurrence): v is integral AND z is real -> polynomial recurrence
 *      Stable for integer degrees, exact polynomial semantics. Works for all real z.
 *    - Path B (Analytic Continuation): non-integer v AND |z| <= 1
 *      Uses T_v(z) = cos(v * acos(z)) with principal branch.
 *    - Path C (Hyperbolic Continuation): non-integer v AND real z with |z| > 1
 *      For z > 1:  T_v(z) = cosh(v * acosh(z))
 *      For z < -1: T_v(z) = cos(v*π) * cosh(v * acosh(-z))
 *    - Path D (Complex): complex v OR complex z -> analytic continuation
 *      Uses T_v(z) = cos(v * acos(z)) with principal branch.
 *
 * 2. BRANCH CONVENTIONS:
 *    - Uses principal branch of acos(z), consistent with PyTorch's complex acos.
 *    - For real z in [-1,1], acos(z) is real in [0, π].
 *    - For real z outside [-1,1] or complex z, acos(z) is complex.
 *    - Branch cuts at z = ±1 follow the standard principal branch definition.
 *
 * 3. DTYPE PROMOTION:
 *    - If either v or z is complex -> output is complex.
 *    - If v is integral and z is floating -> output matches z dtype.
 *    - If both are floating -> output is promoted floating type.
 *    - complex64 if all inputs <= float32/complex64, else complex128.
 *
 * 4. INTEGER VALUE DETECTION:
 *    - Integer values of v are detected using std::floor(v) == v, not by dtype.
 *    - This means v=2.0 (float) uses recurrence, while v=2.5 (float) uses analytic.
 *    - Large integer values (|v| > 2^53) are handled via analytic continuation
 *      to avoid precision loss in int64 conversion.
 *
 * 5. BACKWARD DESIGN:
 *    - Single fused backward kernel returns (gradient_v, gradient_z)
 *    - Single fused double-backward kernel returns (gradient_gradient_output, gradient_v, gradient_z)
 *    - Shares computation of acos(z), sin(v*theta), cos(v*theta), sqrt(1-z²)
 *
 * 6. DERIVATIVE FORMULAS:
 *    For |z| <= 1 (let θ = acos(z)):
 *      ∂T_v(z)/∂z = v * sin(vθ) / sqrt(1 - z²)
 *      ∂T_v(z)/∂v = -sin(vθ) * θ
 *      ∂²T/∂z² = -v² * cos(vθ) / (1-z²) + v * z * sin(vθ) / (1-z²)^(3/2)
 *      ∂²T/∂v² = -cos(vθ) * θ²
 *      ∂²T/∂z∂v = [sin(vθ) + v * θ * cos(vθ)] / sqrt(1-z²)
 *
 *    For z > 1 (let φ = acosh(z)):
 *      ∂T_v(z)/∂z = v * sinh(vφ) / sqrt(z² - 1)
 *      ∂T_v(z)/∂v = sinh(vφ) * φ
 *      ∂²T/∂z² = v² * cosh(vφ) / (z²-1) + v * z * sinh(vφ) / (z²-1)^(3/2)
 *      ∂²T/∂v² = cosh(vφ) * φ²
 *      ∂²T/∂z∂v = [sinh(vφ) + v * φ * cosh(vφ)] / sqrt(z²-1)
 *
 *    For z < -1 (let φ = acosh(-z)):
 *      T_v(z) = cos(vπ) * cosh(vφ)
 *      ∂T_v(z)/∂z = -cos(vπ) * v * sinh(vφ) / sqrt(z² - 1)
 *      ∂T_v(z)/∂v = cos(vπ) * sinh(vφ) * φ - π * sin(vπ) * cosh(vφ)
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <algorithm>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Maximum degree for safe int64_t conversion from floating point
constexpr int64_t kMaxSafeDegree = (1LL << 53);  // 2^53, max exact integer in double

// ============================================================================
// Helper functions
// ============================================================================

/**
 * Robust integer detection that works correctly for all floating-point types.
 *
 * For half-precision types (float16/bfloat16), direct comparison std::floor(v) == v
 * can fail due to rounding errors. This function uses a type-appropriate tolerance
 * to detect integer values reliably.
 *
 * Returns true if v is close enough to an integer to use the recurrence path.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE bool is_integer_value(scalar_t v) {
  using std::abs;
  using std::floor;
  using std::round;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  // Type-appropriate tolerance for integer detection
  // float16: ~1e-3 precision, bfloat16: ~1e-2 precision
  // float32: ~1e-6 precision, float64: ~1e-14 precision
  real_t eps;
  if constexpr (std::is_same_v<real_t, double>) {
    eps = real_t(1e-12);
  } else {
    // For float32, float16, bfloat16 - use a tolerance that accounts for
    // the limited precision. float16 has ~3.3 decimal digits, bfloat16 ~2.4
    eps = real_t(1e-4);
  }

  // Check if v is within tolerance of the nearest integer
  real_t v_real = static_cast<real_t>(v);
  real_t rounded = round(v_real);
  real_t diff = abs(v_real - rounded);

  // For values close to zero, use absolute tolerance
  // For larger values, use relative tolerance
  real_t abs_v = abs(v_real);
  real_t tol = (abs_v > real_t(1)) ? eps * abs_v : eps;

  return diff <= tol && abs_v <= static_cast<real_t>(kMaxSafeDegree);
}

/**
 * Get the integer value from a scalar that has been verified as an integer.
 * Uses round() to handle any residual floating-point error.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE int64_t get_integer_value(scalar_t v) {
  using std::round;
  using std::abs;
  return static_cast<int64_t>(round(abs(static_cast<double>(v))));
}

// ============================================================================
// Forward implementations
// ============================================================================

/**
 * Recurrence-based implementation for integer v and real z.
 * Uses: T_0(z) = 1, T_1(z) = z, T_v(z) = 2z*T_{v-1}(z) - T_{v-2}(z)
 * Note: T_{-n}(z) = T_n(z) for integer n (symmetry property)
 *
 * Small degrees (0-7) are explicitly unrolled for performance. These degrees
 * are common in practice (e.g., spectral methods, polynomial approximation)
 * and benefit from avoiding loop overhead and enabling better instruction
 * scheduling.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t chebyshev_polynomial_t_recurrence(int64_t v, scalar_t z) {
  if (v < 0) v = -v;

  // Explicit unrolling for small degrees (common in practice)
  // These use Horner's form where beneficial for numerical stability
  switch (v) {
    case 0: return scalar_t(1);
    case 1: return z;
    case 2: return scalar_t(2) * z * z - scalar_t(1);
    case 3: return z * (scalar_t(4) * z * z - scalar_t(3));
    case 4: {
      scalar_t z2 = z * z;
      return scalar_t(8) * z2 * (z2 - scalar_t(1)) + scalar_t(1);
    }
    case 5: {
      scalar_t z2 = z * z;
      return z * (z2 * (scalar_t(16) * z2 - scalar_t(20)) + scalar_t(5));
    }
    case 6: {
      scalar_t z2 = z * z;
      return z2 * (z2 * (scalar_t(32) * z2 - scalar_t(48)) + scalar_t(18)) - scalar_t(1);
    }
    case 7: {
      scalar_t z2 = z * z;
      return z * (z2 * (z2 * (scalar_t(64) * z2 - scalar_t(112)) + scalar_t(56)) - scalar_t(7));
    }
    default: break;
  }

  // General recurrence for v >= 8
  // Start from T_6 and T_7 to continue the recurrence
  scalar_t z2 = z * z;
  scalar_t t_prev2 = z2 * (z2 * (scalar_t(32) * z2 - scalar_t(48)) + scalar_t(18)) - scalar_t(1);  // T_6
  scalar_t t_prev1 = z * (z2 * (z2 * (scalar_t(64) * z2 - scalar_t(112)) + scalar_t(56)) - scalar_t(7));  // T_7
  scalar_t t_curr = t_prev1;

  scalar_t two_z = scalar_t(2) * z;

  // Hint for loop unrolling (portable across compilers)
  #if defined(__CUDA_ARCH__)
    #pragma unroll 4
  #elif defined(__clang__)
    #pragma clang loop unroll_count(4)
  #elif defined(__GNUC__)
    #pragma GCC unroll 4
  #endif
  for (int64_t k = 8; k <= v; k++) {
    t_curr = two_z * t_prev1 - t_prev2;
    t_prev2 = t_prev1;
    t_prev1 = t_curr;
  }

  return t_curr;
}

/**
 * Analytic continuation for |z| <= 1: T_v(z) = cos(v * acos(z))
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t chebyshev_polynomial_t_analytic(scalar_t v, scalar_t z) {
  using std::acos;
  using std::cos;
  return cos(v * acos(z));
}

/**
 * Hyperbolic continuation for real z outside [-1, 1]:
 *   For z > 1:  T_v(z) = cosh(v * acosh(z))
 *   For z < -1: T_v(z) = cosh(v * acosh(-z)) * cos(v * π)
 *
 * This follows from the identity T_v(z) = cos(v * acos(z)) where for real z > 1,
 * acos(z) = i * acosh(z), giving cos(i * v * acosh(z)) = cosh(v * acosh(z)).
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t chebyshev_polynomial_t_hyperbolic(scalar_t v, scalar_t z) {
  using std::acosh;
  using std::cosh;
  using std::cos;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t pi = real_t(3.14159265358979323846);

  if (z > scalar_t(1)) {
    return cosh(v * acosh(z));
  } else {
    // z < -1: use T_v(z) = cos(v*π) * T_v(-z)
    return cosh(v * acosh(-z)) * cos(v * pi);
  }
}

/**
 * Main forward function - dispatches to appropriate implementation.
 *
 * Dispatch logic:
 *   - Complex v or z: analytic continuation via cos(v * acos(z))
 *   - Real, integer v: recurrence relation (works for all real z)
 *   - Real, non-integer v, |z| <= 1: analytic continuation via cos(v * acos(z))
 *   - Real, non-integer v, |z| > 1: hyperbolic continuation via cosh(v * acosh(|z|))
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t chebyshev_polynomial_t(scalar_t v, scalar_t z) {
  if constexpr (c10::is_complex<scalar_t>::value) {
    return chebyshev_polynomial_t_analytic(v, z);
  } else {
    if (is_integer_value(v)) {
      // Recurrence works for all real z, not just [-1, 1]
      return chebyshev_polynomial_t_recurrence(get_integer_value(v), z);
    }
    // Non-integer v: check if z is outside [-1, 1]
    if (z > scalar_t(1) || z < scalar_t(-1)) {
      return chebyshev_polynomial_t_hyperbolic(v, z);
    }
    return chebyshev_polynomial_t_analytic(v, z);
  }
}

// ============================================================================
// Fused backward implementation (first-order derivatives)
// ============================================================================

/**
 * Chebyshev U polynomial recurrence for computing dT_n/dz = n * U_{n-1}(z).
 *
 * Small degrees (0-6) are explicitly unrolled. The formulas use Horner's form
 * for improved numerical stability and reduced operation count.
 *
 * U_n(z) polynomials:
 *   U_0(z) = 1
 *   U_1(z) = 2z
 *   U_2(z) = 4z² - 1
 *   U_3(z) = 8z³ - 4z = 4z(2z² - 1)
 *   U_4(z) = 16z⁴ - 12z² + 1
 *   U_5(z) = 32z⁵ - 32z³ + 6z = 2z(16z⁴ - 16z² + 3)
 *   U_6(z) = 64z⁶ - 80z⁴ + 24z² - 1
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t chebyshev_u_recurrence(int64_t n_minus_1, scalar_t z) {
  if (n_minus_1 < 0) return scalar_t(0);

  // Explicit unrolling for small degrees
  switch (n_minus_1) {
    case 0: return scalar_t(1);
    case 1: return scalar_t(2) * z;
    case 2: return scalar_t(4) * z * z - scalar_t(1);
    case 3: return scalar_t(4) * z * (scalar_t(2) * z * z - scalar_t(1));
    case 4: {
      scalar_t z2 = z * z;
      return z2 * (scalar_t(16) * z2 - scalar_t(12)) + scalar_t(1);
    }
    case 5: {
      scalar_t z2 = z * z;
      return scalar_t(2) * z * (z2 * (scalar_t(16) * z2 - scalar_t(16)) + scalar_t(3));
    }
    case 6: {
      scalar_t z2 = z * z;
      return z2 * (z2 * (scalar_t(64) * z2 - scalar_t(80)) + scalar_t(24)) - scalar_t(1);
    }
    default: break;
  }

  // General recurrence for n_minus_1 >= 7
  // Start from U_5 and U_6 to continue the recurrence
  scalar_t z2 = z * z;
  scalar_t u_prev2 = scalar_t(2) * z * (z2 * (scalar_t(16) * z2 - scalar_t(16)) + scalar_t(3));  // U_5
  scalar_t u_prev1 = z2 * (z2 * (scalar_t(64) * z2 - scalar_t(80)) + scalar_t(24)) - scalar_t(1);  // U_6
  scalar_t u_curr = u_prev1;

  scalar_t two_z = scalar_t(2) * z;

  // Hint for loop unrolling (portable across compilers)
  #if defined(__CUDA_ARCH__)
    #pragma unroll 4
  #elif defined(__clang__)
    #pragma clang loop unroll_count(4)
  #elif defined(__GNUC__)
    #pragma GCC unroll 4
  #endif
  for (int64_t k = 7; k <= n_minus_1; k++) {
    u_curr = two_z * u_prev1 - u_prev2;
    u_prev2 = u_prev1;
    u_prev1 = u_curr;
  }

  return u_curr;
}

/**
 * Fused backward - computes both gradient_v and gradient_z in a single pass.
 * Returns: (gradient_v, gradient_z)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t>
chebyshev_polynomial_t_backward(scalar_t grad, scalar_t v, scalar_t z) {
  using std::abs;
  using std::acos;
  using std::cos;
  using std::sin;
  using std::sqrt;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  scalar_t gradient_v = scalar_t(0);
  scalar_t gradient_z = scalar_t(0);

  // Use const instead of constexpr for BFloat16/Half compatibility
  const real_t eps = std::is_same_v<real_t, float> ? real_t(1e-6) : real_t(1e-12);
  const real_t pi = real_t(3.14159265358979323846);

  if constexpr (c10::is_complex<scalar_t>::value) {
    // Complex: always use analytic formulas
    auto theta = acos(z);
    auto sin_v_theta = sin(v * theta);
    auto sqrt_one_minus_z2 = sqrt(scalar_t(1) - z * z);

    auto dTdz = v * sin_v_theta / sqrt_one_minus_z2;
    gradient_z = grad * std::conj(dTdz);

    auto dTdv = -sin_v_theta * theta;
    gradient_v = grad * std::conj(dTdv);
  } else {
    // Real: check for integer v and boundary cases
    bool v_is_integer = is_integer_value(v);

    if (v_is_integer) {
      int64_t v_int = get_integer_value(v);

      // dT_n/dz = n * U_{n-1}(z) using stable recurrence
      if (v_int == 0) {
        gradient_z = scalar_t(0);
      } else {
        gradient_z = grad * scalar_t(v_int) * chebyshev_u_recurrence(v_int - 1, z);
      }

      // Clamp z to [-1, 1] to avoid domain errors in acos for values slightly outside
      auto z_clamped = std::clamp(z, scalar_t(-1), scalar_t(1));
      auto theta = acos(z_clamped);
      auto sin_v_theta = sin(v * theta);
      gradient_v = grad * (-sin_v_theta * theta);
    } else {
      // Non-integer v: analytic/hyperbolic formulas with boundary handling
      real_t dist_to_plus_one = abs(real_t(1) - z);
      real_t dist_to_minus_one = abs(real_t(1) + z);

      if (v == scalar_t(0)) {
        gradient_z = scalar_t(0);
        gradient_v = scalar_t(0);
      } else if (dist_to_plus_one < eps) {
        // Limit at z = 1
        gradient_z = grad * v * v;
        gradient_v = scalar_t(0);
      } else if (dist_to_minus_one < eps) {
        // Limit at z = -1
        gradient_z = grad * v * v * cos(pi * v);
        gradient_v = grad * (-sin(pi * v) * pi);
      } else if (z > scalar_t(1)) {
        // Hyperbolic continuation for z > 1
        // dT_v/dz = v * sinh(v * acosh(z)) / sqrt(z² - 1)
        // dT_v/dv = sinh(v * acosh(z)) * acosh(z)
        using std::acosh;
        using std::sinh;
        using std::cosh;

        auto phi = acosh(z);
        auto sinh_v_phi = sinh(v * phi);
        auto sqrt_z2_minus_1 = sqrt(z * z - scalar_t(1));

        gradient_z = grad * v * sinh_v_phi / sqrt_z2_minus_1;
        gradient_v = grad * sinh_v_phi * phi;
      } else if (z < scalar_t(-1)) {
        // Hyperbolic continuation for z < -1
        // T_v(z) = cos(v*π) * cosh(v * acosh(-z))
        // dT_v/dz = -cos(v*π) * v * sinh(v * acosh(-z)) / sqrt(z² - 1)
        // dT_v/dv = cos(v*π) * sinh(v * acosh(-z)) * acosh(-z) - π * sin(v*π) * cosh(v * acosh(-z))
        using std::acosh;
        using std::sinh;
        using std::cosh;

        auto phi = acosh(-z);
        auto sinh_v_phi = sinh(v * phi);
        auto cosh_v_phi = cosh(v * phi);
        auto cos_v_pi = cos(pi * v);
        auto sin_v_pi = sin(pi * v);
        auto sqrt_z2_minus_1 = sqrt(z * z - scalar_t(1));

        gradient_z = grad * (-cos_v_pi * v * sinh_v_phi / sqrt_z2_minus_1);
        gradient_v = grad * (cos_v_pi * sinh_v_phi * phi - pi * sin_v_pi * cosh_v_phi);
      } else {
        // Standard analytic continuation for |z| < 1
        auto theta = acos(z);
        auto sin_v_theta = sin(v * theta);
        auto sqrt_one_minus_z2 = sqrt(scalar_t(1) - z * z);

        gradient_z = grad * v * sin_v_theta / sqrt_one_minus_z2;
        gradient_v = grad * (-sin_v_theta * theta);
      }
    }
  }

  return std::make_tuple(gradient_v, gradient_z);
}

// ============================================================================
// Fused double-backward implementation (second-order derivatives)
// ============================================================================

/**
 * Fused double-backward computation.
 *
 * Given:
 *   ggv = gradient w.r.t. gradient_v from first backward
 *   ggz = gradient w.r.t. gradient_z from first backward
 *   gradient_output = original upstream gradient
 *   v, z = original inputs
 *
 * Computes:
 *   gradient_gradient_output = ggz * dT/dz + ggv * dT/dv
 *   gradient_v = ggz * gradient_output * d²T/dzdv + ggv * gradient_output * d²T/dv²
 *   gradient_z = ggz * gradient_output * d²T/dz² + ggv * gradient_output * d²T/dvdz
 *
 * Returns: (gradient_gradient_output, gradient_v, gradient_z)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t>
chebyshev_polynomial_t_backward_backward(
    scalar_t ggv,
    scalar_t ggz,
    scalar_t gradient_output,
    scalar_t v,
    scalar_t z,
    bool has_ggv,
    bool has_ggz
) {
  using std::abs;
  using std::acos;
  using std::cos;
  using std::sin;
  using std::sqrt;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  scalar_t gradient_gradient_output = scalar_t(0);
  scalar_t gradient_v = scalar_t(0);
  scalar_t gradient_z = scalar_t(0);

  if (!has_ggv && !has_ggz) {
    return std::make_tuple(gradient_gradient_output, gradient_v, gradient_z);
  }

  // Use const instead of constexpr for BFloat16/Half compatibility
  const real_t eps = std::is_same_v<real_t, float> ? real_t(1e-6) : real_t(1e-12);
  const real_t pi = real_t(3.14159265358979323846);

  // Handle boundary and hyperbolic cases for real types
  if constexpr (!c10::is_complex<scalar_t>::value) {
    real_t dist_to_plus_one = abs(real_t(1) - z);
    real_t dist_to_minus_one = abs(real_t(1) + z);

    if (dist_to_plus_one < eps) {
      // Limits at z = 1:
      // dT/dz = v², dT/dv = 0
      // d²T/dz² = v²(v²-1)/3, d²T/dv² = 0, d²T/dzdv = 2v
      if (has_ggz) {
        gradient_gradient_output = ggz * v * v;
        gradient_z = ggz * gradient_output * v * v * (v * v - scalar_t(1)) / scalar_t(3);
        gradient_v = ggz * gradient_output * scalar_t(2) * v;
      }
      if (has_ggv) {
        gradient_z = gradient_z + ggv * gradient_output * scalar_t(2) * v;
      }
      return std::make_tuple(gradient_gradient_output, gradient_v, gradient_z);
    }

    if (dist_to_minus_one < eps) {
      // Limits at z = -1
      // At z = -1: θ = π, dT/dz = v² * cos(πv), dT/dv = -sin(πv) * π
      // d²T/dz² = v²(v²-1) * cos(πv) / 3
      // d²T/dv² = -π² * cos(πv)
      // d²T/dzdv = d/dv[v² * cos(πv)] = 2v * cos(πv) - πv² * sin(πv)
      auto cos_pi_v = cos(pi * v);
      auto sin_pi_v = sin(pi * v);
      auto d2Tdzdv = scalar_t(2) * v * cos_pi_v - pi * v * v * sin_pi_v;

      if (has_ggz) {
        gradient_gradient_output = ggz * v * v * cos_pi_v;
        gradient_z = ggz * gradient_output * v * v * (v * v - scalar_t(1)) * cos_pi_v / scalar_t(3);
        gradient_v = ggz * gradient_output * d2Tdzdv;
      }
      if (has_ggv) {
        gradient_gradient_output = gradient_gradient_output + ggv * (-sin_pi_v * pi);
        gradient_v = gradient_v + ggv * gradient_output * (-pi * pi * cos_pi_v);
        gradient_z = gradient_z + ggv * gradient_output * d2Tdzdv;
      }
      return std::make_tuple(gradient_gradient_output, gradient_v, gradient_z);
    }

    // Hyperbolic case for z > 1
    if (z > scalar_t(1)) {
      using std::acosh;
      using std::sinh;
      using std::cosh;

      auto phi = acosh(z);
      auto sinh_v_phi = sinh(v * phi);
      auto cosh_v_phi = cosh(v * phi);
      auto z2_minus_1 = z * z - scalar_t(1);
      auto sqrt_z2_minus_1 = sqrt(z2_minus_1);

      // First derivatives
      scalar_t dTdz = v * sinh_v_phi / sqrt_z2_minus_1;
      scalar_t dTdv = sinh_v_phi * phi;

      // Second derivatives
      // d²T/dz² = v² * cosh(v*φ) / (z² - 1) - v * z * sinh(v*φ) / (z² - 1)^(3/2)
      scalar_t d2Tdz2 = v * v * cosh_v_phi / z2_minus_1
                      - v * z * sinh_v_phi / (z2_minus_1 * sqrt_z2_minus_1);
      scalar_t d2Tdv2 = cosh_v_phi * phi * phi;
      scalar_t d2Tdzdv = (sinh_v_phi + v * phi * cosh_v_phi) / sqrt_z2_minus_1;

      if (has_ggz) {
        gradient_gradient_output = ggz * dTdz;
        gradient_z = ggz * gradient_output * d2Tdz2;
        gradient_v = ggz * gradient_output * d2Tdzdv;
      }
      if (has_ggv) {
        gradient_gradient_output = gradient_gradient_output + ggv * dTdv;
        gradient_v = gradient_v + ggv * gradient_output * d2Tdv2;
        gradient_z = gradient_z + ggv * gradient_output * d2Tdzdv;
      }
      return std::make_tuple(gradient_gradient_output, gradient_v, gradient_z);
    }

    // Hyperbolic case for z < -1
    if (z < scalar_t(-1)) {
      using std::acosh;
      using std::sinh;
      using std::cosh;

      auto phi = acosh(-z);
      auto sinh_v_phi = sinh(v * phi);
      auto cosh_v_phi = cosh(v * phi);
      auto cos_v_pi = cos(pi * v);
      auto sin_v_pi = sin(pi * v);
      auto z2_minus_1 = z * z - scalar_t(1);
      auto sqrt_z2_minus_1 = sqrt(z2_minus_1);

      // First derivatives for T_v(z) = cos(v*π) * cosh(v*φ)
      // dT/dz = -cos(v*π) * v * sinh(v*φ) / sqrt(z² - 1)
      // dT/dv = cos(v*π) * sinh(v*φ) * φ - π * sin(v*π) * cosh(v*φ)
      scalar_t dTdz = -cos_v_pi * v * sinh_v_phi / sqrt_z2_minus_1;
      scalar_t dTdv = cos_v_pi * sinh_v_phi * phi - pi * sin_v_pi * cosh_v_phi;

      // Second derivatives
      // d²T/dz² = cos(v*π) * [v² * cosh(v*φ) / (z² - 1) + v * z * sinh(v*φ) / (z² - 1)^(3/2)]
      // d²T/dv² = cos(v*π) * cosh(v*φ) * (φ² - π²) - 2π * sin(v*π) * φ * sinh(v*φ)
      // d²T/dzdv = [-cos(v*π) * (sinh(v*φ) + v*φ*cosh(v*φ)) + π*sin(v*π)*v*sinh(v*φ)] / sqrt(z²-1)
      scalar_t d2Tdz2 = cos_v_pi * (v * v * cosh_v_phi / z2_minus_1
                      + v * z * sinh_v_phi / (z2_minus_1 * sqrt_z2_minus_1));
      scalar_t d2Tdv2 = cos_v_pi * cosh_v_phi * (phi * phi - pi * pi)
                      - scalar_t(2) * pi * sin_v_pi * phi * sinh_v_phi;
      scalar_t d2Tdzdv = (-cos_v_pi * (sinh_v_phi + v * phi * cosh_v_phi)
                       + pi * sin_v_pi * v * sinh_v_phi) / sqrt_z2_minus_1;

      if (has_ggz) {
        gradient_gradient_output = ggz * dTdz;
        gradient_z = ggz * gradient_output * d2Tdz2;
        gradient_v = ggz * gradient_output * d2Tdzdv;
      }
      if (has_ggv) {
        gradient_gradient_output = gradient_gradient_output + ggv * dTdv;
        gradient_v = gradient_v + ggv * gradient_output * d2Tdv2;
        gradient_z = gradient_z + ggv * gradient_output * d2Tdzdv;
      }
      return std::make_tuple(gradient_gradient_output, gradient_v, gradient_z);
    }
  }

  // Standard case: complex types or real z in [-1, 1]
  scalar_t z_safe = z;
  if constexpr (!c10::is_complex<scalar_t>::value) {
    z_safe = std::clamp(z, scalar_t(-1), scalar_t(1));
  }

  // Shared computations
  auto theta = acos(z_safe);
  auto v_theta = v * theta;
  auto sin_v_theta = sin(v_theta);
  auto cos_v_theta = cos(v_theta);
  auto one_minus_z2 = scalar_t(1) - z_safe * z_safe;

  // General case
  auto sqrt_one_minus_z2 = sqrt(one_minus_z2);

  // First derivatives
  scalar_t dTdz = v * sin_v_theta / sqrt_one_minus_z2;
  scalar_t dTdv = -sin_v_theta * theta;

  // Second derivatives
  scalar_t d2Tdz2 = -v * v * cos_v_theta / one_minus_z2
                  + v * z_safe * sin_v_theta / (one_minus_z2 * sqrt_one_minus_z2);
  scalar_t d2Tdv2 = -cos_v_theta * theta * theta;
  scalar_t d2Tdzdv = (sin_v_theta + v * theta * cos_v_theta) / sqrt_one_minus_z2;

  // Apply Wirtinger derivatives for complex types.
  // For backward_backward with complex inputs:
  // - gradient_gradient_output = gg * dT/dx (no conjugate on first derivative)
  // - gradient_x = conj(gg) * gradient_output * conj(d²T/dx²)
  // This matches PyTorch's complex autograd conventions.
  if constexpr (c10::is_complex<scalar_t>::value) {
    if (has_ggz) {
      gradient_gradient_output = ggz * dTdz;
      gradient_z = std::conj(ggz) * gradient_output * std::conj(d2Tdz2);
      gradient_v = std::conj(ggz) * gradient_output * std::conj(d2Tdzdv);
    }

    if (has_ggv) {
      gradient_gradient_output = gradient_gradient_output + ggv * dTdv;
      gradient_v = gradient_v + std::conj(ggv) * gradient_output * std::conj(d2Tdv2);
      gradient_z = gradient_z + std::conj(ggv) * gradient_output * std::conj(d2Tdzdv);
    }
  } else {
    if (has_ggz) {
      gradient_gradient_output = ggz * dTdz;
      gradient_z = ggz * gradient_output * d2Tdz2;
      gradient_v = ggz * gradient_output * d2Tdzdv;
    }

    if (has_ggv) {
      gradient_gradient_output = gradient_gradient_output + ggv * dTdv;
      gradient_v = gradient_v + ggv * gradient_output * d2Tdv2;
      gradient_z = gradient_z + ggv * gradient_output * d2Tdzdv;
    }
  }

  return std::make_tuple(gradient_gradient_output, gradient_v, gradient_z);
}

}  // namespace torchscience::impl::special_functions
