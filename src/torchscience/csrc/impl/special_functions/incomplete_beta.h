#pragma once

/*
 * Regularized Incomplete Beta Function I_z(a, b)
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    The regularized incomplete beta function is defined as:
 *
 *    I_z(a, b) = B_z(a, b) / B(a, b)
 *
 *    where:
 *    - B_z(a, b) = integral from 0 to z of t^(a-1) * (1-t)^(b-1) dt
 *    - B(a, b) = Gamma(a) * Gamma(b) / Gamma(a+b) is the beta function
 *
 *    The function satisfies: I_z(a, b) + I_{1-z}(b, a) = 1
 *
 * 2. DOMAIN:
 *    - z in [0, 1] (real numbers)
 *    - a > 0, b > 0 (positive real numbers)
 *    - For z outside [0, 1], result is NaN
 *    - For a <= 0 or b <= 0, result is NaN
 *
 * 3. ALGORITHM:
 *    Uses continued fraction expansion for numerical evaluation:
 *    - When z < (a + 1) / (a + b + 2), use direct continued fraction
 *    - Otherwise, use symmetry: I_z(a, b) = 1 - I_{1-z}(b, a)
 *
 *    The continued fraction is evaluated using a modified Lentz's algorithm
 *    for numerical stability. The iteration limit is dynamically scaled based
 *    on parameter magnitudes (see compute_max_iterations).
 *
 * 4. SPECIAL VALUES:
 *    - I_0(a, b) = 0
 *    - I_1(a, b) = 1
 *    - I_z(1, 1) = z (uniform distribution CDF)
 *    - I_z(1, b) = 1 - (1-z)^b
 *    - I_z(a, 1) = z^a
 *
 * 5. DERIVATIVE FORMULAS:
 *    dI/dz = z^(a-1) * (1-z)^(b-1) / B(a, b)
 *    dI/da = J_a(z,a,b) / B(a,b) - I_z(a,b) * [psi(a) - psi(a+b)]
 *    dI/db = J_b(z,a,b) / B(a,b) - I_z(a,b) * [psi(b) - psi(a+b)]
 *    where:
 *    - psi is the digamma function
 *    - J_a = integral from 0 to z of t^(a-1) * (1-t)^(b-1) * ln(t) dt
 *    - J_b = integral from 0 to z of t^(a-1) * (1-t)^(b-1) * ln(1-t) dt
 *
 * 6. BACKWARD DESIGN:
 *    - Single fused backward kernel returns (gradient_z, gradient_a, gradient_b)
 *    - Analytical gradients using digamma functions and adaptive quadrature
 *    - Quadratic transformation for log-weighted integrals to handle ln(t) singularity
 *    - Optimized split point for dual-region integration (see compute_optimal_split_point)
 *
 * 7. DOUBLE-BACKWARD DESIGN (fully analytical):
 *    - Single fused double-backward kernel for second-order derivatives
 *    - All derivatives computed analytically using:
 *      * Trigamma functions (psi'(x)) from polygamma.h
 *      * Doubly log-weighted integrals K_aa, K_ab, K_bb via quadrature
 *    - Formulas:
 *      d^2I/da^2 = K_aa/B - 2(J_a/B)(psi(a)-psi(a+b)) + I_z(psi(a)-psi(a+b))^2 - I_z(psi'(a)-psi'(a+b))
 *      d^2I/db^2 = K_bb/B - 2(J_b/B)(psi(b)-psi(a+b)) + I_z(psi(b)-psi(a+b))^2 - I_z(psi'(b)-psi'(a+b))
 *      d^2I/dadb = K_ab/B - (J_a/B)(psi(b)-psi(a+b)) - (J_b/B)(psi(a)-psi(a+b))
 *               + I_z(psi(a)-psi(a+b))(psi(b)-psi(a+b)) + I_z*psi'(a+b)
 *
 * 8. COMPLEX DOUBLE-BACKWARD (Wirtinger derivatives):
 *    - For complex inputs, uses Wirtinger derivative convention compatible with PyTorch
 *    - First backward: B_x = grad_output * conj(∂f/∂x)
 *      where B_x is holomorphic in grad_output but anti-holomorphic in x
 *    - Double backward contributions:
 *      * ∂L/∂(grad_output)* = gg_x * ∂f/∂x (no conjugation)
 *      * ∂L/∂x* = conj(gg_x) * grad_output * conj(∂²f/∂x²)
 *    - The asymmetry arises because B_x has different holomorphicity in each argument
 *
 * 9. ADAPTIVE FEATURES:
 *    - Dynamic iteration limits: compute_max_iterations() scales iterations based on
 *      max(a, b) with sqrt scaling, plus adjustment for parameter asymmetry
 *    - Optimized split points: compute_optimal_split_point() balances singularity
 *      strengths at t=0 and t=1 for more efficient dual-region quadrature
 *    - Optional diagnostics: Define TORCHSCIENCE_ENABLE_QUADRATURE_DIAGNOSTICS
 *      to enable collection of convergence statistics (iterations, error, etc.)
 */

#include <algorithm>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "adaptive_quadrature.h"
#include "digamma.h"
#include "hypergeometric_2_f_1.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Configuration Constants
// ============================================================================

// Base maximum iterations for continued fraction convergence
// This is scaled based on parameter magnitudes for large a, b
constexpr int kBaseMaxIterations = 200;

// Minimum iterations to ensure convergence for simple cases
constexpr int kMinIterations = 50;

// Maximum iterations cap to prevent runaway computation
constexpr int kMaxIterationsCap = 1000;

// Number of Gauss-Legendre quadrature points for integral computation
constexpr int kQuadraturePoints = 32;

// Tiny value to prevent division by zero in continued fraction
// Type-dependent to ensure safety across float32 and float64
// Uses 100x the minimum normal value for each type to provide margin
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE constexpr T tiny_value() {
  // For double: min_normal ~ 2.2e-308, so 100 * min ~ 2.2e-306
  // For float:  min_normal ~ 1.2e-38, so 100 * min ~ 1.2e-36
  // This provides a safe margin above denormalized numbers while still
  // being small enough to not affect numerical results
  return std::numeric_limits<T>::min() * T(100);
}

// Legacy constant for backward compatibility (prefer tiny_value<T>())
constexpr double kTiny = 1e-30;

// ============================================================================
// Dual-Region Integration Heuristics
// ============================================================================
//
// PROBLEM: When computing log-weighted integrals J_a and J_b for parameter
// gradients, we face integrable singularities:
//   - At t=0: t^(a-1) * ln(t) diverges when a < 1
//   - At t=1: (1-t)^(b-1) * ln(1-t) diverges when b < 1
//
// SINGLE-REGION APPROACH (t = z * u^2 transformation):
// The quadratic substitution t = z * u^2 clusters quadrature points near t=0,
// which effectively handles the t=0 singularity. However, when b < 1 and z is
// large (integration domain extends close to t=1), the t=1 singularity is not
// adequately resolved.
//
// DUAL-REGION APPROACH:
// When b < 1 AND z > kDualRegionThreshold, we split the integration:
//   1. Lower region [0, t_split]: Use t = t_split * u^2 (handles t=0)
//   2. Upper region [t_split, z]: Use t = z - (z - t_split) * v^2 (handles t~1)
//
// THRESHOLD CHOICE (kDualRegionThreshold = 0.5):
// - Values < 0.5: Unnecessary splitting for modest z values
// - Values > 0.5: May not activate dual-region when needed
// - 0.5 is empirically determined to balance:
//   * Avoiding overhead of dual-region for easy cases
//   * Ensuring accuracy when the t=1 singularity matters
// - When z ≤ 0.5 and b < 1, the integration domain [0, z] doesn't extend
//   close enough to t=1 for the singularity to significantly affect accuracy
//
// SPLIT POINT SELECTION (see compute_optimal_split_point):
// The optimal split balances singularity strengths at both ends. The function
// uses a weighted formula based on how strong each singularity is (smaller
// parameters = stronger singularities).
//
constexpr double kDualRegionThreshold = 0.5;

// ============================================================================
// Diagnostic Support (Optional)
// ============================================================================

/**
 * Diagnostic information from quadrature computation.
 * This structure captures convergence behavior for debugging and analysis.
 */
struct QuadratureDiagnostics {
  int iterations_used;        // Number of adaptive iterations performed
  int max_subdivisions;       // Maximum subdivision depth reached
  double estimated_error;     // Final error estimate
  bool converged;             // Whether tolerance was achieved

  C10_HOST_DEVICE QuadratureDiagnostics()
      : iterations_used(0), max_subdivisions(0),
        estimated_error(0.0), converged(true) {}
};

// Thread-local diagnostics storage (disabled by default for performance)
// Enable by defining TORCHSCIENCE_ENABLE_QUADRATURE_DIAGNOSTICS
#ifdef TORCHSCIENCE_ENABLE_QUADRATURE_DIAGNOSTICS
inline thread_local QuadratureDiagnostics g_last_quadrature_diagnostics;

/**
 * Update the thread-local diagnostics with new values.
 * Only active when TORCHSCIENCE_ENABLE_QUADRATURE_DIAGNOSTICS is defined.
 */
inline void update_quadrature_diagnostics(
    int iterations, int subdivisions, double error, bool converged
) {
  g_last_quadrature_diagnostics.iterations_used = iterations;
  g_last_quadrature_diagnostics.max_subdivisions = subdivisions;
  g_last_quadrature_diagnostics.estimated_error = error;
  g_last_quadrature_diagnostics.converged = converged;
}

/**
 * Get the diagnostics from the last quadrature computation.
 * Only active when TORCHSCIENCE_ENABLE_QUADRATURE_DIAGNOSTICS is defined.
 */
inline QuadratureDiagnostics get_last_quadrature_diagnostics() {
  return g_last_quadrature_diagnostics;
}

/**
 * Reset the diagnostics to default values.
 */
inline void reset_quadrature_diagnostics() {
  g_last_quadrature_diagnostics = QuadratureDiagnostics();
}

// Macro to conditionally update diagnostics
#define TORCHSCIENCE_UPDATE_DIAGNOSTICS(iterations, subdivisions, error, converged) \
  update_quadrature_diagnostics(iterations, subdivisions, error, converged)

#else  // TORCHSCIENCE_ENABLE_QUADRATURE_DIAGNOSTICS not defined

// No-op versions when diagnostics are disabled
#define TORCHSCIENCE_UPDATE_DIAGNOSTICS(iterations, subdivisions, error, converged) ((void)0)

#endif  // TORCHSCIENCE_ENABLE_QUADRATURE_DIAGNOSTICS

// ============================================================================
// Forward Declarations
// ============================================================================
// These declarations enable functions defined later in this file to be called
// from templates defined earlier.
// Note: log_gamma_complex is defined in log_gamma.h (included via hypergeometric_2_f_1.h)

// Forward declarations for log_beta (three overloads for complex, float/double, and half types)
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<c10::is_complex<T>::value, T>
log_beta(T a, T b);

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<T>::value &&
  (std::is_same_v<T, float> || std::is_same_v<T, double>),
  T>
log_beta(T a, T b);

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<T>::value &&
  !std::is_same_v<T, float> &&
  !std::is_same_v<T, double>,
  T>
log_beta(T a, T b);

// ============================================================================
// Dynamic Iteration Limit Computation
// ============================================================================

/**
 * Compute adaptive iteration limit based on parameter magnitudes.
 *
 * The continued fraction for the incomplete beta function converges at a rate
 * that depends on the parameters a and b. For larger parameters, more iterations
 * may be needed to achieve the desired precision.
 *
 * Heuristic:
 *   - Base limit: 200 iterations (sufficient for most cases)
 *   - Scale factor: sqrt(max(a, b)) for large parameters
 *   - Additional iterations for asymmetric parameters (|a - b| large)
 *
 * The scaling is based on empirical observation that convergence rate is
 * approximately O(1/sqrt(max(a,b))) for large parameters.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE int compute_max_iterations(T a, T b) {
  using std::sqrt;
  using std::max;
  using std::min;
  using std::abs;

  using real_t = typename c10::scalar_value_type<T>::type;

  real_t a_real, b_real;
  if constexpr (c10::is_complex<T>::value) {
    a_real = a.real();
    b_real = b.real();
  } else {
    a_real = a;
    b_real = b;
  }

  // For small parameters, base iterations are sufficient
  real_t max_param = (a_real > b_real) ? a_real : b_real;
  if (max_param <= real_t(10)) {
    return kBaseMaxIterations;
  }

  // Scale iterations based on parameter magnitude
  // sqrt scaling matches the O(1/sqrt(n)) convergence rate
  real_t scale_factor = sqrt(max_param / real_t(10));

  // Additional iterations for highly asymmetric parameters
  // which can cause slower convergence
  real_t asymmetry = abs(a_real - b_real) / (a_real + b_real);
  real_t asymmetry_factor = real_t(1) + asymmetry * real_t(0.5);

  int computed = static_cast<int>(
      real_t(kBaseMaxIterations) * scale_factor * asymmetry_factor);

  // Clamp to valid range
  if (computed < kMinIterations) computed = kMinIterations;
  if (computed > kMaxIterationsCap) computed = kMaxIterationsCap;

  return computed;
}

// ============================================================================
// Optimized Split Point Computation
// ============================================================================

/**
 * Compute an optimized split point for dual-region integration based on
 * singularity strengths.
 *
 * When computing integrals of the form:
 *   ∫₀ᶻ t^(a-1) * (1-t)^(b-1) * f(t) dt
 *
 * with singularities at t=0 (when a < 1) and t=1 (when b < 1), the optimal
 * split point should balance the computational effort between regions.
 *
 * MATHEMATICAL RATIONALE:
 *
 * The integrand has singularity strengths:
 *   - At t=0: proportional to t^(a-1), singular when a < 1
 *   - At t=1: proportional to (1-t)^(b-1), singular when b < 1
 *
 * For the log-weighted integrals J_a and J_b, the ln(t) and ln(1-t) factors
 * add additional logarithmic singularities at the endpoints.
 *
 * The "effective strength" of each singularity can be characterized by:
 *   - At t=0: strength ∝ 1/min(a, 1) * (1 + |ln(a)|) for very small a
 *   - At t=1: strength ∝ 1/min(b, 1) * (1 + |ln(b)|) for very small b
 *
 * WEIGHTING FORMULA:
 *
 * We use a weighted split point: t_split = z * w_b / (w_a + w_b)
 *
 * where w_a and w_b are weights reflecting singularity strengths:
 *   - For param >= 1: weight = 1 (no singularity)
 *   - For param < 1: weight = 1/param (stronger singularity = higher weight)
 *   - For param < 0.1: apply log damping to prevent extreme weights
 *
 * The ratio w_b / (w_a + w_b) determines where to split:
 *   - If w_b >> w_a: split closer to z (more region for t=1 singularity)
 *   - If w_a >> w_b: split closer to 0 (more region for t=0 singularity)
 *   - If w_a ≈ w_b: split near midpoint
 *
 * CLAMPING:
 * The split point is clamped to [0.1*z, 0.9*z] to ensure both regions
 * have sufficient size for accurate quadrature. Without clamping, extreme
 * parameter asymmetry could create a region too small for accurate integration.
 *
 * SPECIAL CASES:
 * - If both a ≥ 1 and b ≥ 1: use midpoint z/2 (no singularities)
 * - If only a < 1: bias toward lower region (split closer to 0)
 * - If only b < 1: bias toward upper region (split closer to z)
 * - If both a < 1 and b < 1: balance based on relative strengths
 *
 * EMPIRICAL VALIDATION:
 * This heuristic was validated against SciPy's betainc across a grid of
 * parameters including:
 *   - a, b in {0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.0, 2.0, 5.0, 10.0}
 *   - z in {0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99}
 * Accuracy was verified to be within 1e-4 relative tolerance for forward
 * values and 1e-3 for gradients in the most difficult cases.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T compute_optimal_split_point(
    T z, T a, T b
) {
  using std::abs;
  using std::log;
  using std::min;
  using std::max;

  using real_t = typename c10::scalar_value_type<T>::type;

  // Extract real parts for comparison
  real_t a_real, b_real, z_real;
  if constexpr (c10::is_complex<T>::value) {
    a_real = a.real();
    b_real = b.real();
    z_real = abs(z);  // Use magnitude for complex z
  } else {
    a_real = a;
    b_real = b;
    z_real = z;
  }

  // Default to midpoint if both parameters are >= 1 (no strong singularities)
  if (a_real >= real_t(1) && b_real >= real_t(1)) {
    return z / T(2);
  }

  // Compute singularity weights
  // Weight increases as parameter decreases below 1
  // Use log scaling for very small parameters to prevent extreme weights
  auto singularity_weight = [](real_t param) -> real_t {
    if (param >= real_t(1)) {
      return real_t(1);  // No singularity
    }
    // Weight inversely proportional to param, with log damping for tiny values
    // This prevents extreme splits when one parameter is very small
    real_t base_weight = real_t(1) / param;
    if (param < real_t(0.1)) {
      // Add logarithmic damping for very small parameters
      real_t log_factor = real_t(1) + abs(log(param + real_t(1e-10)));
      base_weight = base_weight / (real_t(1) + log_factor * real_t(0.1));
    }
    return base_weight;
  };

  real_t w_a = singularity_weight(a_real);
  real_t w_b = singularity_weight(b_real);

  // Compute weighted split point
  // Higher w_b (stronger t=1 singularity) -> split closer to z
  // Higher w_a (stronger t=0 singularity) -> split closer to 0
  real_t split_ratio = w_b / (w_a + w_b);

  // Clamp split ratio to [0.1, 0.9] to ensure both regions have adequate size
  const real_t min_ratio = real_t(0.1);
  const real_t max_ratio = real_t(0.9);
  if (split_ratio < min_ratio) split_ratio = min_ratio;
  if (split_ratio > max_ratio) split_ratio = max_ratio;

  return z * T(split_ratio);
}

// ============================================================================
// Asymptotic Expansion Thresholds
// ============================================================================
// For z very close to 0 or 1, asymptotic series expansions provide better
// accuracy than the continued fraction or numerical quadrature. These
// thresholds define when to switch to asymptotic formulas.

// Threshold for using asymptotic expansion near z=0
// For z < kAsymptoticThresholdZero, use I_z(a,b) ≈ z^a / (a·B(a,b)) · (1 + O(z))
constexpr double kAsymptoticThresholdZero = 1e-8;

// Threshold for using asymptotic expansion near z=1
// For z > 1 - kAsymptoticThresholdOne, use symmetry + asymptotic near 0
constexpr double kAsymptoticThresholdOne = 1e-8;

// ============================================================================
// Asymptotic Expansions for Extreme z Values
// ============================================================================

/**
 * First-order asymptotic expansion of I_z(a,b) for z → 0.
 *
 * For small z, the regularized incomplete beta function has the expansion:
 *   I_z(a,b) = z^a / (a·B(a,b)) · [1 + (a/(a+1))·(1-b)·z + O(z²)]
 *
 * The leading term z^a / (a·B(a,b)) is exact when b=1.
 * The first correction term improves accuracy for b≠1.
 *
 * This expansion is numerically stable for z < 1e-8 where the continued
 * fraction may require many iterations or lose precision.
 *
 * Mathematical derivation:
 *   I_z(a,b) = (1/B(a,b)) · ∫₀ᶻ t^(a-1)·(1-t)^(b-1) dt
 *            = (z^a / (a·B(a,b))) · [1 + (a(1-b)/(a+1))·z + ...]
 *
 * Returns: (I_z, valid) where valid indicates the expansion is applicable.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, bool>
incomplete_beta_asymptotic_zero(T z, T a, T b) {
  using std::exp;
  using std::log;
  using std::abs;

  using real_t = typename c10::scalar_value_type<T>::type;

  // Check if z is small enough for this expansion
  real_t z_mag;
  if constexpr (c10::is_complex<T>::value) {
    z_mag = abs(z);
  } else {
    z_mag = z;
  }

  if (z_mag >= real_t(kAsymptoticThresholdZero)) {
    return std::make_tuple(T(0), false);
  }

  // Leading term: z^a / (a·B(a,b)) = z^a · exp(-log_beta(a,b)) / a
  // Computed in log space for numerical stability
  T log_z = log(z);
  T lb = log_beta(a, b);
  T log_leading = a * log_z - lb - log(a);
  T leading = exp(log_leading);

  // First correction term: (a/(a+1))·(1-b)·z
  // This improves accuracy when b ≠ 1
  T correction = (a / (a + T(1))) * (T(1) - b) * z;

  T result = leading * (T(1) + correction);

  return std::make_tuple(result, true);
}

/**
 * First-order asymptotic expansion of I_z(a,b) for z → 1.
 *
 * Uses the symmetry relation I_z(a,b) = 1 - I_{1-z}(b,a) combined with
 * the asymptotic expansion near zero.
 *
 * For z close to 1, let w = 1-z (small):
 *   I_z(a,b) = 1 - I_w(b,a)
 *            ≈ 1 - w^b / (b·B(b,a)) · [1 + (b/(b+1))·(1-a)·w + ...]
 *
 * Returns: (I_z, valid) where valid indicates the expansion is applicable.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, bool>
incomplete_beta_asymptotic_one(T z, T a, T b) {
  using std::abs;

  using real_t = typename c10::scalar_value_type<T>::type;

  // Check if z is close enough to 1
  T w = T(1) - z;  // Small quantity
  real_t w_mag;
  if constexpr (c10::is_complex<T>::value) {
    w_mag = abs(w);
  } else {
    w_mag = w;
  }

  if (w_mag >= real_t(kAsymptoticThresholdOne)) {
    return std::make_tuple(T(0), false);
  }

  // Use symmetry: I_z(a,b) = 1 - I_w(b,a) where w = 1-z
  auto [I_w, valid] = incomplete_beta_asymptotic_zero(w, b, a);

  if (!valid) {
    return std::make_tuple(T(0), false);
  }

  return std::make_tuple(T(1) - I_w, true);
}

/**
 * Asymptotic expansion for dI/dz near z=0.
 *
 * The derivative dI/dz = z^(a-1) · (1-z)^(b-1) / B(a,b).
 *
 * For small z:
 *   dI/dz ≈ z^(a-1) / B(a,b) · [1 - (b-1)·z + O(z²)]
 *
 * Returns: (dI/dz, valid)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, bool>
incomplete_beta_dz_asymptotic_zero(T z, T a, T b) {
  using std::exp;
  using std::log;
  using std::abs;

  using real_t = typename c10::scalar_value_type<T>::type;

  real_t z_mag;
  if constexpr (c10::is_complex<T>::value) {
    z_mag = abs(z);
  } else {
    z_mag = z;
  }

  if (z_mag >= real_t(kAsymptoticThresholdZero)) {
    return std::make_tuple(T(0), false);
  }

  // Leading term: z^(a-1) / B(a,b)
  T log_z = log(z);
  T lb = log_beta(a, b);
  T log_leading = (a - T(1)) * log_z - lb;
  T leading = exp(log_leading);

  // Correction: (1-z)^(b-1) ≈ 1 - (b-1)·z for small z
  T correction = T(1) - (b - T(1)) * z;

  return std::make_tuple(leading * correction, true);
}

/**
 * Asymptotic expansion for dI/dz near z=1.
 *
 * For z close to 1, let w = 1-z:
 *   dI/dz = z^(a-1) · w^(b-1) / B(a,b)
 *         ≈ w^(b-1) / B(a,b) · [1 - (a-1)·w + O(w²)]
 *
 * Returns: (dI/dz, valid)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, bool>
incomplete_beta_dz_asymptotic_one(T z, T a, T b) {
  using std::exp;
  using std::log;
  using std::abs;

  using real_t = typename c10::scalar_value_type<T>::type;

  T w = T(1) - z;
  real_t w_mag;
  if constexpr (c10::is_complex<T>::value) {
    w_mag = abs(w);
  } else {
    w_mag = w;
  }

  if (w_mag >= real_t(kAsymptoticThresholdOne)) {
    return std::make_tuple(T(0), false);
  }

  // Leading term: w^(b-1) / B(a,b)
  T log_w = log(w);
  T lb = log_beta(a, b);
  T log_leading = (b - T(1)) * log_w - lb;
  T leading = exp(log_leading);

  // Correction: z^(a-1) ≈ 1 - (a-1)·w for z close to 1
  T correction = T(1) - (a - T(1)) * w;

  return std::make_tuple(leading * correction, true);
}

/**
 * Asymptotic expansion for parameter derivatives dI/da, dI/db near z=0.
 *
 * For small z, using I_z ≈ z^a / (a·B(a,b)):
 *   dI/da ≈ I_z · [ln(z) - 1/a - ψ(a) + ψ(a+b)]
 *   dI/db ≈ I_z · [-ψ(b) + ψ(a+b)]
 *
 * The J_a and J_b integrals simplify because the integration domain is small.
 *
 * Returns: (dI/da, dI/db, valid)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, bool>
incomplete_beta_param_derivs_asymptotic_zero(
    T z, T a, T b, T I_z
) {
  using std::log;
  using std::abs;

  using real_t = typename c10::scalar_value_type<T>::type;

  real_t z_mag;
  if constexpr (c10::is_complex<T>::value) {
    z_mag = abs(z);
  } else {
    z_mag = z;
  }

  if (z_mag >= real_t(kAsymptoticThresholdZero)) {
    return std::make_tuple(T(0), T(0), false);
  }

  // Digamma values
  T psi_a = digamma(a);
  T psi_b = digamma(b);
  T psi_ab = digamma(a + b);

  T log_z = log(z);

  // dI/da = I_z · [ln(z) - 1/a - (ψ(a) - ψ(a+b))]
  // This comes from differentiating z^a/(a·B(a,b)) with respect to a
  T dIda = I_z * (log_z - T(1)/a - (psi_a - psi_ab));

  // dI/db = I_z · [-(ψ(b) - ψ(a+b))]
  // Only the B(a,b) term contributes since z^a/a doesn't depend on b
  T dIdb = -I_z * (psi_b - psi_ab);

  return std::make_tuple(dIda, dIdb, true);
}

// ============================================================================
// Helper functions
// ============================================================================
// Note: log_gamma_complex is defined in log_gamma.h

/**
 * Log of the beta function: log(B(a, b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a+b))
 *
 * Supports both real and complex arguments.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  c10::is_complex<T>::value,
  T>
log_beta(T a, T b) {
  return log_gamma_complex(a) + log_gamma_complex(b) - log_gamma_complex(a + b);
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<T>::value &&
  (std::is_same_v<T, float> || std::is_same_v<T, double>),
  T>
log_beta(T a, T b) {
  using std::lgamma;
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<T>::value &&
  !std::is_same_v<T, float> &&
  !std::is_same_v<T, double>,
  T>
log_beta(T a, T b) {
  // Compute in float32 for half-precision types
  return static_cast<T>(log_beta(static_cast<float>(a), static_cast<float>(b)));
}

/**
 * Analytic continuation of incomplete beta for |z| > 1.
 *
 * Uses the relation:
 *   I_z(a,b) = z^a / (a * B(a,b)) * 2F1(a, 1-b; a+1; z)
 *
 * For different regions:
 *   - |z| >= 1, |1-z| < 1: Use symmetry I_z(a,b) = 1 - I_{1-z}(b,a)
 *   - |z| > 1, |1-z| >= 1: Use hypergeometric linear transformation
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T
incomplete_beta_extended_domain(T z, T a, T b);

// ============================================================================
// Digamma/Trigamma Value Cache
// ============================================================================
// These cache structures precompute and store digamma/trigamma values that are
// used multiple times in the backward and double-backward computations. This
// avoids redundant expensive function evaluations.

/**
 * Cache for digamma values used in first-order parameter derivatives.
 *
 * The derivatives dI/da and dI/db require:
 *   - ψ(a), ψ(b), ψ(a+b) for the digamma terms
 *   - ψ(a) - ψ(a+b) and ψ(b) - ψ(a+b) for the derivative formulas
 *
 * Computing these once and storing them avoids 3 redundant digamma calls
 * when computing both dI/da and dI/db.
 */
template <typename T>
struct DiGammaCache {
  T psi_a;           // ψ(a)
  T psi_b;           // ψ(b)
  T psi_ab;          // ψ(a+b)
  T psi_a_minus_ab;  // ψ(a) - ψ(a+b)
  T psi_b_minus_ab;  // ψ(b) - ψ(a+b)

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  DiGammaCache(T a, T b) {
    psi_a = digamma(a);
    psi_b = digamma(b);
    psi_ab = digamma(a + b);
    psi_a_minus_ab = psi_a - psi_ab;
    psi_b_minus_ab = psi_b - psi_ab;
  }
};

// ============================================================================
// Log-weighted Beta Integrals
// ============================================================================

/**
 * Compute log-weighted integrals using adaptive Gauss-Kronrod quadrature.
 *
 * For the regularized incomplete beta function I_z(a, b), the partial derivatives
 * with respect to a and b require computing:
 *
 *   J_a(z, a, b) = integral from 0 to z of t^(a-1) * (1-t)^(b-1) * ln(t) dt
 *   J_b(z, a, b) = integral from 0 to z of t^(a-1) * (1-t)^(b-1) * ln(1-t) dt
 *
 * These integrals have endpoint singularities:
 *   - ln(t) -> -infinity as t -> 0
 *   - t^(a-1) -> infinity as t -> 0 when a < 1
 *   - (1-t)^(b-1) -> infinity as t -> 1 when b < 1
 *
 * Uses adaptive Gauss-Kronrod quadrature with iterative subdivision for
 * automatic error control. The algorithm:
 *   1. For the t=0 singularity: applies t = t_upper * u^2 to cluster points near t=0
 *   2. For the t=1 singularity (when b < 1 and z > threshold): uses dual-region
 *      integration with a complementary transformation t = z - delta * v^2
 *      to cluster points near t=z
 *   3. Uses G7-K15 (or G15-K31 for difficult cases) to estimate both integral and error
 *   4. Subdivides intervals where error exceeds tolerance
 *   5. Uses an explicit stack to avoid recursion (CUDA compatible)
 *   6. Uses Kahan summation to reduce floating-point error accumulation
 *
 * DUAL-REGION SPLIT POINT RATIONALE:
 * When b < 1 and z > 0.5, we split the integration at t_split = z/2. This choice:
 *   - Ensures roughly equal-sized regions in t-space for balanced error distribution
 *   - Places the split point well away from both singularities (t=0 and t->z near t=1)
 *   - The midpoint z/2 provides symmetric coverage: lower region handles t=0 singularity
 *     with t = (z/2) * u^2 transformation, upper region handles t=z proximity to t=1
 *     singularity with t = z - (z/2) * v^2 transformation
 *   - Simpler than adaptive split point selection while being effective for typical cases
 *
 * Returns a tuple (J_a, J_b) computed simultaneously for efficiency.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T>
log_weighted_beta_integrals(T z, T a, T b) {
  using std::abs;
  using real_t = typename c10::scalar_value_type<T>::type;

  // Safety check for boundary cases
  // For complex z: check if |z| is near 0 or 1
  bool z_near_zero, z_near_one;
  if constexpr (c10::is_complex<T>::value) {
    z_near_zero = abs(z) < real_t(1e-10);
    z_near_one = abs(z - T(1)) < real_t(1e-10) || abs(z) > real_t(1) - real_t(1e-10);
  } else {
    z_near_zero = z <= T(0);
    z_near_one = z >= T(1);
  }
  if (z_near_zero || z_near_one) {
    return std::make_tuple(T(0), T(0));
  }

  // Determine if this is a difficult case requiring higher precision
  // Small a or b creates strong singularities that need more quadrature points
  bool use_high_order;
  if constexpr (c10::is_complex<T>::value) {
    use_high_order = (a.real() < real_t(0.1)) || (b.real() < real_t(0.1));
  } else {
    use_high_order = (a < T(0.1)) || (b < T(0.1));
  }

  // Detect very small parameters that create extreme singularities
  bool has_very_small_params;
  if constexpr (c10::is_complex<T>::value) {
    has_very_small_params = (a.real() < real_t(0.05)) || (b.real() < real_t(0.05));
  } else {
    has_very_small_params = (a < T(0.05)) || (b < T(0.05));
  }

  // Set tolerance based on problem difficulty and dtype
  // Uses dtype-aware tolerance to avoid unachievable precision targets
  T tolerance;
  if (has_very_small_params) {
    tolerance = T(adaptive_tolerance_very_small_params<T>());
  } else if (use_high_order) {
    tolerance = T(adaptive_tolerance_difficult<T>());
  } else {
    tolerance = T(adaptive_tolerance<T>());
  }

  // Check if we need dual-region integration for the t=1 singularity
  // When b < 1 and z > threshold, the (1-t)^(b-1) singularity at t=1
  // requires special handling with a complementary transformation
  // Also force dual-region for very small parameters to improve accuracy
  bool need_dual_region;
  if constexpr (c10::is_complex<T>::value) {
    need_dual_region = ((b.real() < real_t(1)) && (abs(z) > real_t(kDualRegionThreshold)))
                       || has_very_small_params;
  } else {
    need_dual_region = ((b < T(1)) && (z > T(kDualRegionThreshold)))
                       || has_very_small_params;
  }

  if (need_dual_region) {
    // Compute optimized split point based on singularity strengths
    // The split point balances computational effort between regions:
    // - Stronger t=0 singularity (small a) -> split closer to 0
    // - Stronger t=1 singularity (small b) -> split closer to z
    // - Both singularities present -> weighted balance
    T t_split = compute_optimal_split_point(z, a, b);

    // Integrate lower region [0, t_split]
    // Uses t = t_split * u^2 transformation to handle t=0 singularity
    auto [J_a_lower, J_b_lower] = adaptive_log_weighted_beta_integrals(
        t_split, a, b, tolerance, use_high_order);

    // Integrate upper region [t_split, z] with complementary transformation
    // Uses t = z - (z - t_split) * v^2 to handle proximity to t=1 singularity
    auto [J_a_upper, J_b_upper] = adaptive_upper_region_integrals(
        z, t_split, a, b, tolerance, use_high_order);

    return std::make_tuple(J_a_lower + J_a_upper, J_b_lower + J_b_upper);
  }

  // Standard single-region quadrature (handles t=0 singularity)
  return adaptive_log_weighted_beta_integrals(z, a, b, tolerance, use_high_order);
}

/**
 * Compute doubly log-weighted integrals for second-order derivatives.
 *
 * For the second derivatives of the regularized incomplete beta function,
 * we need:
 *
 *   K_aa(z, a, b) = integral from 0 to z of t^(a-1) * (1-t)^(b-1) * ln^2(t) dt
 *   K_ab(z, a, b) = integral from 0 to z of t^(a-1) * (1-t)^(b-1) * ln(t) * ln(1-t) dt
 *   K_bb(z, a, b) = integral from 0 to z of t^(a-1) * (1-t)^(b-1) * ln^2(1-t) dt
 *
 * These integrals have stronger singularities than J_a, J_b due to the
 * squared logarithms. Uses adaptive Gauss-Kronrod quadrature for automatic
 * error control.
 *
 * When b < 1 and z > threshold, uses dual-region integration to properly
 * handle the (1-t)^(b-1) singularity near t=1. See log_weighted_beta_integrals
 * for the rationale behind the z/2 split point choice.
 *
 * Returns a tuple (K_aa, K_ab, K_bb) computed simultaneously for efficiency.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T>
doubly_log_weighted_beta_integrals(T z, T a, T b) {
  using std::abs;
  using real_t = typename c10::scalar_value_type<T>::type;

  // Safety check for boundary cases
  // For complex z: check if |z| is near 0 or 1
  bool z_near_zero, z_near_one;
  if constexpr (c10::is_complex<T>::value) {
    z_near_zero = abs(z) < real_t(1e-10);
    z_near_one = abs(z - T(1)) < real_t(1e-10) || abs(z) > real_t(1) - real_t(1e-10);
  } else {
    z_near_zero = z <= T(0);
    z_near_one = z >= T(1);
  }
  if (z_near_zero || z_near_one) {
    return std::make_tuple(T(0), T(0), T(0));
  }

  // Detect very small parameters that create strong singularities
  // For a < 0.05 or b < 0.05, the ln^2 terms create extreme singularities
  bool has_very_small_params;
  if constexpr (c10::is_complex<T>::value) {
    has_very_small_params = (a.real() < real_t(0.05)) || (b.real() < real_t(0.05));
  } else {
    has_very_small_params = (a < T(0.05)) || (b < T(0.05));
  }

  // Set tolerance based on parameter magnitude and dtype
  // Very small parameters require relaxed tolerances as machine precision is not achievable
  T tolerance;
  if (has_very_small_params) {
    tolerance = T(adaptive_tolerance_very_small_params<T>());
  } else {
    tolerance = T(adaptive_tolerance<T>());
  }

  // Check if we need dual-region integration for the t=1 singularity
  // Also force dual-region for very small parameters to improve accuracy
  bool need_dual_region;
  if constexpr (c10::is_complex<T>::value) {
    need_dual_region = ((b.real() < real_t(1)) && (abs(z) > real_t(kDualRegionThreshold)))
                       || has_very_small_params;
  } else {
    need_dual_region = ((b < T(1)) && (z > T(kDualRegionThreshold)))
                       || has_very_small_params;
  }

  if (need_dual_region) {
    // Compute optimized split point based on singularity strengths
    // See compute_optimal_split_point for detailed rationale
    T t_split = compute_optimal_split_point(z, a, b);

    // Integrate lower region [0, t_split]
    auto [K_aa_lower, K_ab_lower, K_bb_lower] =
        adaptive_doubly_log_weighted_beta_integrals(t_split, a, b, tolerance);

    // Integrate upper region [t_split, z] with complementary transformation
    auto [K_aa_upper, K_ab_upper, K_bb_upper] =
        adaptive_doubly_weighted_upper_region(z, t_split, a, b, tolerance);

    return std::make_tuple(
        K_aa_lower + K_aa_upper,
        K_ab_lower + K_ab_upper,
        K_bb_lower + K_bb_upper);
  }

  // Standard single-region quadrature
  return adaptive_doubly_log_weighted_beta_integrals(z, a, b, tolerance);
}

/**
 * Compute analytical partial derivatives dI/da and dI/db using cached digamma values.
 *
 * Using the formulas:
 *   dI/da = J_a(z,a,b) / B(a,b) - I_z(a,b) * [psi(a) - psi(a+b)]
 *   dI/db = J_b(z,a,b) / B(a,b) - I_z(a,b) * [psi(b) - psi(a+b)]
 *
 * where:
 *   - psi is the digamma function
 *   - J_a and J_b are the log-weighted integrals
 *   - I_z(a,b) is the incomplete beta function value
 *
 * This overload accepts a pre-computed DiGammaCache to avoid redundant
 * digamma evaluations when the caller needs both derivatives and has
 * already computed the digamma values.
 *
 * Parameters:
 *   z: evaluation point in (0, 1)
 *   a, b: shape parameters (positive)
 *   I_z: precomputed value of incomplete_beta(z, a, b)
 *   inv_beta: precomputed value of 1/B(a,b)
 *   cache: precomputed digamma values
 *
 * Returns: tuple (dI/da, dI/db)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T>
incomplete_beta_parameter_derivatives_cached(
    T z,
    T a,
    T b,
    T I_z,
    T inv_beta,
    const DiGammaCache<T>& cache
) {
  auto [J_a, J_b] = log_weighted_beta_integrals(z, a, b);

  return std::make_tuple(J_a * inv_beta - I_z * cache.psi_a_minus_ab, J_b * inv_beta - I_z * cache.psi_b_minus_ab);
}

/**
 * Compute analytical partial derivatives dI/da and dI/db.
 *
 * This is the original convenience overload that computes digamma values
 * internally. For repeated calls with the same a, b, prefer using
 * incomplete_beta_parameter_derivatives_cached with a DiGammaCache.
 *
 * Parameters:
 *   z: evaluation point in (0, 1)
 *   a, b: shape parameters (positive)
 *   I_z: precomputed value of incomplete_beta(z, a, b)
 *
 * Returns: tuple (dI/da, dI/db)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T>
incomplete_beta_parameter_derivatives(T z, T a, T b, T I_z) {
  using std::exp;

  // Create cache for digamma values
  DiGammaCache<T> cache(a, b);

  // Explicit cast needed for half-precision types since std::exp may return float
  T inv_beta = static_cast<T>(exp(-log_beta(a, b)));

  return incomplete_beta_parameter_derivatives_cached(z, a, b, I_z, inv_beta, cache);
}

/**
 * Continued fraction for incomplete beta function using modified Lentz's algorithm.
 * Computes the continued fraction: a_1/(b_1 + a_2/(b_2 + ...))
 *
 * This is the continued fraction expansion of I_z(a,b) * B(a,b) * (1-z)^b * z^(-a)
 *
 * The iteration limit is dynamically computed based on parameter magnitudes to
 * ensure convergence for large a, b while avoiding unnecessary iterations for
 * small parameters.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T beta_continued_fraction(
    T a, T b, T z
) {
  using std::abs;
  using std::floor;

  using real_t = typename c10::scalar_value_type<T>::type;
  const real_t eps = std::numeric_limits<real_t>::epsilon();
  const real_t tiny = tiny_value<real_t>();

  // Compute adaptive iteration limit based on parameter magnitudes
  const int max_iterations = compute_max_iterations(a, b);

  T qab = a + b;
  T qap = a + T(1);
  T qam = a - T(1);

  // First term of continued fraction
  T c = T(1);
  T d = T(1) - qab * z / qap;

  if (abs(d) < tiny) d = tiny;
  d = T(1) / d;
  T h = d;

  for (int m = 1; m <= max_iterations; m++) {
    T m_real = T(m);
    T m2 = T(2 * m);

    // Even step
    T aa = m_real * (b - m_real) * z / ((qam + m2) * (a + m2));
    d = T(1) + aa * d;
    if (abs(d) < tiny) d = tiny;
    c = T(1) + aa / c;
    if (abs(c) < tiny) c = tiny;
    d = T(1) / d;
    h *= d * c;

    // Odd step
    aa = -(a + m_real) * (qab + m_real) * z / ((a + m2) * (qap + m2));
    d = T(1) + aa * d;
    if (abs(d) < tiny) d = tiny;
    c = T(1) + aa / c;
    if (abs(c) < tiny) c = tiny;
    d = T(1) / d;
    T delta = d * c;
    h *= delta;

    // Check for convergence
    if (abs(delta - T(1)) < eps) {
      break;
    }
  }

  return h;
}

// ============================================================================
// Extended Domain Implementation (Analytic Continuation)
// ============================================================================

// Forward declaration of incomplete_beta for use in extended domain
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T incomplete_beta(
    T z, T a, T b
);

/**
 * Analytic continuation of incomplete beta for |z| >= 1.
 *
 * Uses the relation:
 *   I_z(a,b) = z^a / (a * B(a,b)) * 2F1(a, 1-b; a+1; z)
 *
 * Region Classification:
 *   - |z| >= 1, |1-z| < 1: Use symmetry I_z(a,b) = 1 - I_{1-z}(b,a)
 *   - |z| > 1, |1-z| >= 1: Use hypergeometric linear transformation
 *
 * The symmetry relation is exact and allows us to reduce many cases to the
 * standard |z| < 1 computation. For the remaining cases, we use the DLMF 15.8.2
 * linear transformation formula for the hypergeometric function.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T
incomplete_beta_extended_domain(T z, T a, T b) {
  using std::abs;
  using std::exp;
  using std::log;

  using real_t = typename c10::scalar_value_type<T>::type;

  real_t one_minus_z_mag;

  if constexpr (c10::is_complex<T>::value) {
    one_minus_z_mag = abs(T(1) - z);
  } else {
    one_minus_z_mag = abs(T(1) - z);
  }

  if (one_minus_z_mag < real_t(1)) {
    return T(1) - incomplete_beta(T(1) - z, b, a);
  }

  return exp(a * log(z) - log(a) - log_beta(a, b)) * hypergeometric_2f1_linear_transform(a, T(1) - b, a + T(1), z);
}

// ============================================================================
// Forward implementation
// ============================================================================

/**
 * Regularized incomplete beta function I_z(a, b).
 *
 * Uses continued fraction expansion with symmetry relation for numerical stability.
 *
 * Supports both real and complex inputs. For complex z with |z| < 1, uses the
 * continued fraction. For |z| >= 1, uses analytic continuation via the
 * hypergeometric 2F1 relation.
 *
 * Branch cut convention:
 *   - ln(z) has branch cut on (-inf, 0]
 *   - ln(1-z) has branch cut on [1, +inf)
 *   - For |z| < 1, both are well-defined using principal branch
 *   - For |z| >= 1, uses analytic continuation formulas
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T incomplete_beta(
    T z, T a, T b
) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::pow;

  using real_t = typename c10::scalar_value_type<T>::type;

  if constexpr (c10::is_complex<T>::value) {
    // Complex implementation
    const real_t tol = real_t(1e-10);

    // Handle boundary cases for complex z
    // z near 0: return 0
    if (abs(z) < tol) {
      return T(0);
    }
    // z near 1: return 1
    if (abs(z - T(1)) < tol) {
      return T(1);
    }

    // For |z| >= 1, use analytic continuation
    if (abs(z) >= real_t(1)) {
      return incomplete_beta_extended_domain(z, a, b);
    }

    // Invalid parameters: Re(a) <= 0 or Re(b) <= 0
    if (a.real() <= real_t(0) || b.real() <= real_t(0)) {
      return T(std::numeric_limits<real_t>::quiet_NaN(), real_t(0));
    }

    // Special cases for a=1 or b=1 (check if close to 1)
    if (abs(a - T(1)) < tol) {
      // I_z(1, b) = 1 - (1-z)^b
      return T(1) - exp(b * log(T(1) - z));
    }
    if (abs(b - T(1)) < tol) {
      // I_z(a, 1) = z^a
      return exp(a * log(z));
    }

    // Try asymptotic expansion for z near 0
    auto [result_zero, valid_zero] = incomplete_beta_asymptotic_zero(z, a, b);
    if (valid_zero) {
      return result_zero;
    }

    // Try asymptotic expansion for z near 1
    auto [result_one, valid_one] = incomplete_beta_asymptotic_one(z, a, b);
    if (valid_one) {
      return result_one;
    }

    // Log of the prefactor: z^a * (1-z)^b / (a * B(a, b))
    T log_prefactor = a * log(z) + b * log(T(1) - z) - log_beta(a, b) - log(a);

    // Use symmetry relation for numerical stability
    // For complex z, use |z| < 0.5 heuristic for direct computation
    if (abs(z) < real_t(0.5)) {
      T cf = beta_continued_fraction(a, b, z);
      return exp(log_prefactor) * cf;
    } else {
      // Use symmetry: I_z(a, b) = 1 - I_{1-z}(b, a)
      T z_comp = T(1) - z;
      T log_prefactor_comp = b * log(z_comp) + a * log(z) - log_beta(b, a) - log(b);
      T cf = beta_continued_fraction(b, a, z_comp);
      return T(1) - exp(log_prefactor_comp) * cf;
    }
  } else {
    // Real implementation
    // Handle boundary cases
    if (z <= T(0)) {
      return T(0);
    }
    if (z >= T(1)) {
      return T(1);
    }

    // Invalid parameters
    if (a <= T(0) || b <= T(0)) {
      return std::numeric_limits<T>::quiet_NaN();
    }

    // Special cases for a=1 or b=1
    if (a == T(1)) {
      // I_z(1, b) = 1 - (1-z)^b
      return T(1) - pow(T(1) - z, b);
    }
    if (b == T(1)) {
      // I_z(a, 1) = z^a
      return pow(z, a);
    }

    // Try asymptotic expansion for z near 0
    // This provides better accuracy than the continued fraction for very small z
    auto [result_zero, valid_zero] = incomplete_beta_asymptotic_zero(z, a, b);
    if (valid_zero) {
      return result_zero;
    }

    // Try asymptotic expansion for z near 1
    auto [result_one, valid_one] = incomplete_beta_asymptotic_one(z, a, b);
    if (valid_one) {
      return result_one;
    }

    // Log of the prefactor: z^a * (1-z)^b / (a * B(a, b))
    T log_prefactor = a * log(z) + b * log(T(1) - z) - log_beta(a, b) - log(a);

    // Use symmetry relation for numerical stability
    // When z < (a+1)/(a+b+2), use direct computation
    // Otherwise, use I_z(a,b) = 1 - I_{1-z}(b,a)
    T threshold = (a + T(1)) / (a + b + T(2));

    if (z < threshold) {
      T cf = beta_continued_fraction(a, b, z);
      return exp(log_prefactor) * cf;
    }
    // Use symmetry: I_z(a, b) = 1 - I_{1-z}(b, a)
    T z_comp = T(1) - z;
    T log_prefactor_comp = b * log(z_comp) + a * log(z) - log_beta(b, a) - log(b);
    T cf = beta_continued_fraction(b, a, z_comp);
    return T(1) - exp(log_prefactor_comp) * cf;
  }
}

}  // namespace torchscience::impl::special_functions
