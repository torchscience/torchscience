#pragma once

/*
 * Chebyshev Polynomial of the First Kind T_v(z)
 *
 * DESIGN NOTES:
 *
 * 1. DISPATCH LOGIC:
 *    - Path A (Recurrence): v is integral AND z is real -> polynomial recurrence
 *      Stable for integer degrees, exact polynomial semantics.
 *    - Path B (Analytic Continuation): non-integer v OR complex v OR complex z
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
 *    First order:
 *      ∂T_v(z)/∂z = v * sin(v*acos(z)) / sqrt(1 - z^2)
 *      ∂T_v(z)/∂v = -sin(v*acos(z)) * acos(z)
 *    Second order:
 *      ∂²T/∂z² = -v² * cos(vθ) / (1-z²) + v * z * sin(vθ) / (1-z²)^(3/2)
 *      ∂²T/∂v² = -cos(vθ) * θ²
 *      ∂²T/∂z∂v = [sin(vθ) + v * θ * cos(vθ)] / sqrt(1-z²)
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <tuple>
#include <type_traits>

namespace torchscience::impl::special_functions {

// Maximum degree for safe int64_t conversion from floating point
constexpr int64_t kMaxSafeDegree = (1LL << 53);  // 2^53, max exact integer in double

// ============================================================================
// Forward implementations
// ============================================================================

/**
 * Recurrence-based implementation for integer v and real z.
 * Uses: T_0(z) = 1, T_1(z) = z, T_v(z) = 2z*T_{v-1}(z) - T_{v-2}(z)
 * Note: T_{-n}(z) = T_n(z) for integer n (symmetry property)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t chebyshev_polynomial_t_recurrence(int64_t v, scalar_t z) {
  if (v < 0) v = -v;
  if (v == 0) return scalar_t(1);
  if (v == 1) return z;

  scalar_t t_prev2 = scalar_t(1);
  scalar_t t_prev1 = z;
  scalar_t t_curr = t_prev1;

  for (int64_t k = 2; k <= v; k++) {
    t_curr = scalar_t(2) * z * t_prev1 - t_prev2;
    t_prev2 = t_prev1;
    t_prev1 = t_curr;
  }

  return t_curr;
}

/**
 * Analytic continuation: T_v(z) = cos(v * acos(z))
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t chebyshev_polynomial_t_analytic(scalar_t v, scalar_t z) {
  using std::acos;
  using std::cos;
  return cos(v * acos(z));
}

/**
 * Main forward function - dispatches to appropriate implementation.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t chebyshev_polynomial_t(scalar_t v, scalar_t z) {
  if constexpr (c10::is_complex<scalar_t>::value) {
    return chebyshev_polynomial_t_analytic(v, z);
  } else {
    if (std::floor(v) == v && std::abs(v) <= static_cast<scalar_t>(kMaxSafeDegree)) {
      return chebyshev_polynomial_t_recurrence(static_cast<int64_t>(v), z);
    }
    return chebyshev_polynomial_t_analytic(v, z);
  }
}

// ============================================================================
// Fused backward implementation (first-order derivatives)
// ============================================================================

/**
 * Chebyshev U polynomial recurrence for computing dT_n/dz = n * U_{n-1}(z).
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t chebyshev_u_recurrence(int64_t n_minus_1, scalar_t z) {
  if (n_minus_1 < 0) return scalar_t(0);
  if (n_minus_1 == 0) return scalar_t(1);
  if (n_minus_1 == 1) return scalar_t(2) * z;

  scalar_t u_prev2 = scalar_t(1);
  scalar_t u_prev1 = scalar_t(2) * z;
  scalar_t u_curr = u_prev1;

  for (int64_t k = 2; k <= n_minus_1; k++) {
    u_curr = scalar_t(2) * z * u_prev1 - u_prev2;
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
    bool v_is_integer = (std::floor(v) == v && abs(v) <= static_cast<scalar_t>(kMaxSafeDegree));

    if (v_is_integer) {
      int64_t v_int = static_cast<int64_t>(abs(v));

      // dT_n/dz = n * U_{n-1}(z) using stable recurrence
      if (v_int == 0) {
        gradient_z = scalar_t(0);
      } else {
        gradient_z = grad * scalar_t(v_int) * chebyshev_u_recurrence(v_int - 1, z);
      }

      auto theta = acos(z);
      auto sin_v_theta = sin(v * theta);
      gradient_v = grad * (-sin_v_theta * theta);
    } else {
      // Non-integer v: analytic formulas with boundary handling
      real_t dist_to_plus_one = abs(real_t(1) - z);
      real_t dist_to_minus_one = abs(real_t(1) + z);

      if (v == scalar_t(0)) {
        gradient_z = scalar_t(0);
        gradient_v = scalar_t(0);
      } else if (dist_to_plus_one < eps) {
        gradient_z = grad * v * v;
        gradient_v = scalar_t(0);
      } else if (dist_to_minus_one < eps) {
        gradient_z = grad * v * v * cos(pi * v);
        gradient_v = grad * (-sin(pi * v) * pi);
      } else {
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

  // Shared computations
  auto theta = acos(z);
  auto v_theta = v * theta;
  auto sin_v_theta = sin(v_theta);
  auto cos_v_theta = cos(v_theta);
  auto one_minus_z2 = scalar_t(1) - z * z;

  // Handle boundary cases for real types
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
      auto cos_pi_v = cos(pi * v);
      auto sin_pi_v = sin(pi * v);

      if (has_ggz) {
        gradient_gradient_output = ggz * v * v * cos_pi_v;
        gradient_z = ggz * gradient_output * v * v * (v * v - scalar_t(1)) * cos_pi_v / scalar_t(3);
        auto d2Tdzdv_numer = sin_pi_v + v * pi * cos_pi_v;
        gradient_v = ggz * gradient_output * d2Tdzdv_numer / sqrt(one_minus_z2);
      }
      if (has_ggv) {
        gradient_gradient_output = gradient_gradient_output + ggv * (-sin_pi_v * pi);
        gradient_v = gradient_v + ggv * gradient_output * (-pi * pi * cos_pi_v);
        auto d2Tdvdz_numer = sin_pi_v + v * pi * cos_pi_v;
        gradient_z = gradient_z + ggv * gradient_output * d2Tdvdz_numer / sqrt(one_minus_z2);
      }
      return std::make_tuple(gradient_gradient_output, gradient_v, gradient_z);
    }
  }

  // General case
  auto sqrt_one_minus_z2 = sqrt(one_minus_z2);

  // First derivatives
  scalar_t dTdz = v * sin_v_theta / sqrt_one_minus_z2;
  scalar_t dTdv = -sin_v_theta * theta;

  // Second derivatives
  scalar_t d2Tdz2 = -v * v * cos_v_theta / one_minus_z2
                  + v * z * sin_v_theta / (one_minus_z2 * sqrt_one_minus_z2);
  scalar_t d2Tdv2 = -cos_v_theta * theta * theta;
  scalar_t d2Tdzdv = (sin_v_theta + v * theta * cos_v_theta) / sqrt_one_minus_z2;

  // Apply Wirtinger conjugation for complex types
  if constexpr (c10::is_complex<scalar_t>::value) {
    dTdz = std::conj(dTdz);
    dTdv = std::conj(dTdv);
    d2Tdz2 = std::conj(d2Tdz2);
    d2Tdv2 = std::conj(d2Tdv2);
    d2Tdzdv = std::conj(d2Tdzdv);
  }

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

}  // namespace torchscience::impl::special_functions
