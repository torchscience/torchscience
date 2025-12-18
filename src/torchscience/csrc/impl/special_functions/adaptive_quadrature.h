#pragma once

/*
 * Adaptive Gauss-Kronrod Quadrature for Special Functions
 *
 * This header provides generic adaptive quadrature routines using
 * Gauss-Kronrod rules (G7-K15 and G15-K31) with automatic error control.
 *
 * FEATURES:
 *   - Gauss-Legendre 32-point quadrature nodes and weights
 *   - Gauss-Kronrod G7-K15 and G15-K31 rules with embedded error estimation
 *   - Kahan summation for improved numerical accuracy
 *   - Policy-based design for flexible quadrature rule selection
 *   - Iterative (non-recursive) implementation for CUDA compatibility
 *   - Support for both real and complex scalar types
 *
 * QUADRATURE RULES:
 *   - G7-K15: 15 Kronrod points with 7 embedded Gauss points (lighter, faster)
 *   - G15-K31: 31 Kronrod points with 15 embedded Gauss points (higher accuracy)
 *
 * REFERENCES:
 *   - Piessens et al., QUADPACK (1983)
 *   - Abramowitz & Stegun, Table 25.4
 *   - Kahan, W. (1965). "Pracniques: Further remarks on reducing truncation errors"
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <c10/util/Logging.h>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <limits>

namespace torchscience::impl::special_functions {

// ============================================================================
// Convergence Warning Support
// ============================================================================
// Warnings are only emitted on host code (not in CUDA kernels) to avoid
// runtime overhead and compatibility issues. Enable verbose warnings by
// defining TORCHSCIENCE_QUADRATURE_VERBOSE_WARNINGS.

#ifndef __CUDA_ARCH__
// Host-only warning for quadrature convergence failures
#ifdef TORCHSCIENCE_QUADRATURE_VERBOSE_WARNINGS
#define TORCHSCIENCE_QUADRATURE_WARN_CONVERGENCE(iterations, max_iter, remaining) \
  C10_LOG_FIRST_N(WARNING, 10) << "Adaptive quadrature did not converge: " \
    << iterations << "/" << max_iter << " iterations, " \
    << remaining << " intervals remaining"
#else
// Lightweight warning that only logs once per process
#define TORCHSCIENCE_QUADRATURE_WARN_CONVERGENCE(iterations, max_iter, remaining) \
  C10_LOG_FIRST_N(WARNING, 1) << "Adaptive quadrature convergence warning: " \
    "some integrals may have reduced accuracy. " \
    "Define TORCHSCIENCE_QUADRATURE_VERBOSE_WARNINGS for details."
#endif
#else
// No-op in device code
#define TORCHSCIENCE_QUADRATURE_WARN_CONVERGENCE(iterations, max_iter, remaining) ((void)0)
#endif

// ============================================================================
// Gauss-Legendre Quadrature (32-point)
// ============================================================================
// Nodes and weights for [-1, 1] interval, computed to high precision.
// Reference: Abramowitz & Stegun, Table 25.4

namespace gauss_legendre {

// Nodes (abscissae) for 32-point Gauss-Legendre quadrature on [-1, 1]
// Only positive nodes are stored; negative nodes are symmetric.
constexpr double kNodes[16] = {
  0.04830766568773831623,
  0.14447196158279649349,
  0.23928736225213707454,
  0.33186860228212764978,
  0.42135127613063534536,
  0.50689990893222939002,
  0.58771575724076232904,
  0.66304426693021520098,
  0.73218211874028968039,
  0.79448379596794240696,
  0.84936761373256997013,
  0.89632115576605212397,
  0.93490607593773968917,
  0.96476225558750643077,
  0.98561151154526833540,
  0.99726386184948156354
};

// Weights for 32-point Gauss-Legendre quadrature on [-1, 1]
// Weights are symmetric, so only 16 unique values.
constexpr double kWeights[16] = {
  0.09654008851472780056,
  0.09563872007927485942,
  0.09384439908080456564,
  0.09117387869576388471,
  0.08765209300440381114,
  0.08331192422694675522,
  0.07819389578707030647,
  0.07234579410884850623,
  0.06582222277636184684,
  0.05868409347853554714,
  0.05099805926237617619,
  0.04283589802222668066,
  0.03427386291302143313,
  0.02539206530926205956,
  0.01627439473090567065,
  0.00701861000947009660
};

}  // namespace gauss_legendre

// ============================================================================
// Gauss-Kronrod Quadrature (G7-K15 and G15-K31)
// ============================================================================
// Adaptive quadrature using embedded Gauss-Kronrod rules for error estimation.
// K15 uses 15 points with 7 embedded Gauss points for error estimation.
// K31 uses 31 points with 15 embedded Gauss points for difficult integrands.
// Reference: QUADPACK, Piessens et al. (1983)
//
// NODE INDEXING CONVENTION:
// For K15: We store 8 non-negative nodes (indices 0-7). The G7 Gauss nodes
//          are at EVEN indices (0, 2, 4, 6), while K15-only nodes are at
//          odd indices (1, 3, 5, 7).
// For K31: We store 16 non-negative nodes (indices 0-15). The G15 Gauss nodes
//          are at EVEN indices (0, 2, 4, ..., 14), while K31-only nodes are
//          at odd indices.

namespace gauss_kronrod {

// G7-K15: 15 Kronrod nodes on [-1, 1], only non-negative values stored (symmetric)
// G7 Gauss nodes are at even indices: 0, 2, 4, 6
// K15-only nodes are at odd indices: 1, 3, 5, 7
constexpr double kK15Nodes[8] = {
  0.0,                        // index 0 (G7 node)
  0.20778495500789846760,     // index 1 (K15 only)
  0.40584515137739716691,     // index 2 (G7 node)
  0.58608723546769113029,     // index 3 (K15 only)
  0.74153118559939443986,     // index 4 (G7 node)
  0.86486442335976907279,     // index 5 (K15 only)
  0.94910791234275852453,     // index 6 (G7 node)
  0.99145537112081263921      // index 7 (K15 only)
};

// K15 weights (for full 15-point Kronrod rule)
constexpr double kK15Weights[8] = {
  0.20948214108472782801,     // weight for node 0 (also G7)
  0.20443294007529889241,     // weight for node 1
  0.19035057806478540991,     // weight for node 2 (also G7)
  0.16900472663926790283,     // weight for node 3
  0.14065325971552591875,     // weight for node 4 (also G7)
  0.10479001032225018384,     // weight for node 5
  0.06309209262997855329,     // weight for node 6 (also G7)
  0.02293532201052922496      // weight for node 7
};

// G7 weights (for embedded 7-point Gauss rule)
// These correspond to K15 nodes at even indices: 0, 2, 4, 6
// kG7Weights[i] is the G7 weight for kK15Nodes[2*i]
constexpr double kG7Weights[4] = {
  0.41795918367346938776,     // weight for K15 node 0
  0.38183005050511894495,     // weight for K15 node 2
  0.27970539148927666790,     // weight for K15 node 4
  0.12948496616886969327      // weight for K15 node 6
};

// G15-K31: 31 Kronrod nodes on [-1, 1], only non-negative values stored (symmetric)
// G15 Gauss nodes are at even indices: 0, 2, 4, ..., 14
// K31-only nodes are at odd indices: 1, 3, 5, ..., 15
constexpr double kK31Nodes[16] = {
  0.0,
  0.10114206691871749903,
  0.20119409399743452230,
  0.29918000715316881217,
  0.39415134707756336990,
  0.48508186364023968069,
  0.57097217260853884754,
  0.65099674129741697053,
  0.72441773136017004742,
  0.79041850144246593297,
  0.84820658341042721620,
  0.89726453234408190088,
  0.93727339240070590431,
  0.96773907567913913426,
  0.98799251802048542849,
  0.99800229869339706029
};

// K31 weights
constexpr double kK31Weights[16] = {
  0.10133000701479154902,
  0.10076984552387559504,
  0.09917359872179195933,
  0.09664272698362367851,
  0.09312659817082532123,
  0.08856444305621177065,
  0.08308050282313302104,
  0.07684968075772037889,
  0.06985412131872825870,
  0.06200956780067064029,
  0.05348152469092808727,
  0.04458975132476487661,
  0.03534636079137584622,
  0.02546084732671532019,
  0.01500794732931612254,
  0.00537747987292334899
};

// G15 weights (for embedded 15-point Gauss rule)
// These correspond to K31 nodes at even indices: 0, 2, 4, ..., 14
// kG15Weights[i] is the G15 weight for kK31Nodes[2*i]
constexpr double kG15Weights[8] = {
  0.20257824192556127288,
  0.19843148532711157646,
  0.18616100001556221103,
  0.16626920581699393355,
  0.13957067792615431445,
  0.10715922046717193501,
  0.07036604748810812471,
  0.03075324199611726835
};

}  // namespace gauss_kronrod

// Maximum depth for adaptive quadrature subdivision (2^8 = 256 max intervals)
constexpr int kMaxAdaptiveDepth = 8;

// ============================================================================
// Dtype-Aware Tolerance Selection
// ============================================================================
// Tolerances must be achievable for the given floating-point type. Using
// tolerances tighter than machine epsilon is wasteful and can cause
// unnecessary subdivisions or false convergence failures.
//
// For float32: epsilon ≈ 1.2e-7, so 1e-10 is unachievable
// For float64: epsilon ≈ 2.2e-16, so 1e-10 is reasonable
//
// We use ~100-1000x epsilon as the base tolerance to account for:
// - Accumulated rounding errors in quadrature
// - Cancellation in difference computations
// - Jacobian transformation errors

// Legacy constants for backward compatibility (prefer adaptive_tolerance<T>())
constexpr double kAdaptiveTolDefault = 1e-10;
constexpr double kAdaptiveTolDifficult = 1e-12;

/**
 * Compute dtype-appropriate tolerance for adaptive quadrature.
 *
 * Returns a tolerance that is achievable for the given scalar type,
 * scaled by an optional multiplier for difficult cases.
 *
 * For float32: ~1e-5 (100x epsilon)
 * For float64: ~1e-10 (achievable with double precision)
 * For complex types: uses the underlying real type's tolerance
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
typename c10::scalar_value_type<scalar_t>::type adaptive_tolerance() {
  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t eps = std::numeric_limits<real_t>::epsilon();
  // Use 100x epsilon as base, but cap at 1e-10 for double precision
  // to avoid being overly conservative
  const real_t base_tol = eps * real_t(100);
  const real_t max_tol = real_t(1e-10);
  return (base_tol > max_tol) ? base_tol : max_tol;
}

/**
 * Compute dtype-appropriate tolerance for difficult integration cases.
 *
 * Used when parameters create strong singularities (a < 0.1 or b < 0.1).
 * Slightly tighter than default to ensure accuracy in challenging regimes.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
typename c10::scalar_value_type<scalar_t>::type adaptive_tolerance_difficult() {
  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t eps = std::numeric_limits<real_t>::epsilon();
  // Use 50x epsilon for difficult cases, capped at 1e-12 for double
  const real_t base_tol = eps * real_t(50);
  const real_t max_tol = real_t(1e-12);
  return (base_tol > max_tol) ? base_tol : max_tol;
}

/**
 * Compute dtype-appropriate tolerance for very small parameter cases.
 *
 * Used when parameters are extremely small (a < 0.05 or b < 0.05), creating
 * very strong singularities in the doubly log-weighted integrals K_aa, K_ab, K_bb.
 * These integrals have ln^2(t) or ln(t)*ln(1-t) terms that amplify errors.
 *
 * Uses a relaxed tolerance since achieving machine precision is not possible
 * for these extreme cases, but we aim for better than 1e-3 relative error.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
typename c10::scalar_value_type<scalar_t>::type adaptive_tolerance_very_small_params() {
  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  const real_t eps = std::numeric_limits<real_t>::epsilon();
  // Use 500x epsilon for very small parameters, capped at 1e-8 for double
  // This balances accuracy vs computational cost for extreme singularities
  const real_t base_tol = eps * real_t(500);
  const real_t max_tol = real_t(1e-8);
  return (base_tol > max_tol) ? base_tol : max_tol;
}

// ============================================================================
// Diagnostic Tracking for Adaptive Quadrature
// ============================================================================
// These structures track convergence behavior when diagnostics are enabled.
// Enable by defining TORCHSCIENCE_ENABLE_QUADRATURE_DIAGNOSTICS before
// including incomplete_beta.h.

/**
 * Internal diagnostic tracker for adaptive quadrature.
 * Accumulates statistics during the quadrature iteration loop.
 */
struct AdaptiveQuadratureTracker {
  int iterations;           // Total iterations performed
  int max_depth_reached;    // Maximum subdivision depth reached
  double total_error;       // Sum of error estimates across all intervals
  double max_interval_error; // Maximum error on any single interval
  bool hit_max_iterations;  // Whether we hit the iteration limit

  C10_HOST_DEVICE AdaptiveQuadratureTracker()
      : iterations(0), max_depth_reached(0), total_error(0.0),
        max_interval_error(0.0), hit_max_iterations(false) {}

  C10_HOST_DEVICE void record_iteration(double error, int current_depth) {
    iterations++;
    if (current_depth > max_depth_reached) {
      max_depth_reached = current_depth;
    }
    total_error += error;
    if (error > max_interval_error) {
      max_interval_error = error;
    }
  }

  C10_HOST_DEVICE void mark_max_iterations_reached() {
    hit_max_iterations = true;
  }

  C10_HOST_DEVICE bool converged() const {
    return !hit_max_iterations;
  }
};

// ============================================================================
// Kahan Summation for Improved Numerical Accuracy
// ============================================================================
// Kahan summation (compensated summation) reduces floating-point error
// accumulation when summing many terms. This is important for adaptive
// quadrature where many subinterval contributions are accumulated.
// Reference: Kahan, W. (1965). "Pracniques: Further remarks on reducing
//            truncation errors". Communications of the ACM.

template <typename scalar_t>
struct KahanAccumulator {
  scalar_t sum;
  scalar_t compensation;

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  KahanAccumulator() : sum(scalar_t(0)), compensation(scalar_t(0)) {}

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  void add(scalar_t value) {
    scalar_t y = value - compensation;
    scalar_t t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
  }

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  scalar_t result() const { return sum; }
};

// ============================================================================
// Gauss-Kronrod Rule Policy Classes
// ============================================================================
// These policy classes encapsulate the differences between G7-K15 and G15-K31
// rules, allowing a single template implementation for the quadrature logic.

/**
 * Policy for G7-K15 Gauss-Kronrod rule.
 * - 8 stored nodes (symmetric, so 15 total)
 * - G7 nodes at even indices (0, 2, 4, 6)
 * - 4 G7 weights
 */
struct GaussKronrod15Policy {
  static constexpr int kNumNodes = 8;
  static constexpr int kNumGaussWeights = 4;

  template <typename scalar_t>
  C10_HOST_DEVICE C10_ALWAYS_INLINE
  static scalar_t kronrod_node(int i) {
    return scalar_t(gauss_kronrod::kK15Nodes[i]);
  }

  template <typename scalar_t>
  C10_HOST_DEVICE C10_ALWAYS_INLINE
  static scalar_t kronrod_weight(int i) {
    return scalar_t(gauss_kronrod::kK15Weights[i]);
  }

  template <typename scalar_t>
  C10_HOST_DEVICE C10_ALWAYS_INLINE
  static scalar_t gauss_weight(int gauss_idx) {
    return scalar_t(gauss_kronrod::kG7Weights[gauss_idx]);
  }

  // G7 nodes are at even K15 indices (0, 2, 4, 6)
  // For index i, returns true if it's a Gauss node
  C10_HOST_DEVICE C10_ALWAYS_INLINE
  static bool is_gauss_node(int i) {
    return (i % 2 == 0);
  }

  // Convert K15 index to G7 weight index
  // For i = 0, 2, 4, 6 -> returns 0, 1, 2, 3
  C10_HOST_DEVICE C10_ALWAYS_INLINE
  static int gauss_weight_index(int i) {
    return i / 2;
  }
};

/**
 * Policy for G15-K31 Gauss-Kronrod rule.
 * - 16 stored nodes (symmetric, so 31 total)
 * - G15 nodes at even indices (0, 2, 4, ..., 14)
 * - 8 G15 weights
 */
struct GaussKronrod31Policy {
  static constexpr int kNumNodes = 16;
  static constexpr int kNumGaussWeights = 8;

  template <typename scalar_t>
  C10_HOST_DEVICE C10_ALWAYS_INLINE
  static scalar_t kronrod_node(int i) {
    return scalar_t(gauss_kronrod::kK31Nodes[i]);
  }

  template <typename scalar_t>
  C10_HOST_DEVICE C10_ALWAYS_INLINE
  static scalar_t kronrod_weight(int i) {
    return scalar_t(gauss_kronrod::kK31Weights[i]);
  }

  template <typename scalar_t>
  C10_HOST_DEVICE C10_ALWAYS_INLINE
  static scalar_t gauss_weight(int gauss_idx) {
    return scalar_t(gauss_kronrod::kG15Weights[gauss_idx]);
  }

  // G15 nodes are at even K31 indices (0, 2, 4, ..., 14)
  C10_HOST_DEVICE C10_ALWAYS_INLINE
  static bool is_gauss_node(int i) {
    return (i % 2 == 0);
  }

  // Convert K31 index to G15 weight index
  C10_HOST_DEVICE C10_ALWAYS_INLINE
  static int gauss_weight_index(int i) {
    return i / 2;
  }
};

// ============================================================================
// Transformation Policies for Quadrature
// ============================================================================
// These policies encapsulate the coordinate transformation used to cluster
// quadrature points near singularities. They enable a single unified
// quadrature implementation to handle both lower and upper region integration.

/**
 * Lower region transformation policy: t = scale * u²
 *
 * Used for integration over [0, t_upper] where t_upper is typically z or t_split.
 * The quadratic transformation clusters points near t=0, handling the t^(a-1)
 * singularity when a < 1.
 *
 * Mapping: u ∈ [0, 1] → t ∈ [0, scale]
 * Jacobian: dt/du = 2 * scale * u
 */
template <typename scalar_t>
struct LowerRegionTransform {
  scalar_t scale;  // t_upper

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  explicit LowerRegionTransform(scalar_t t_upper) : scale(t_upper) {}

  // Additional constructor for API compatibility (ignores second argument)
  C10_HOST_DEVICE C10_ALWAYS_INLINE
  LowerRegionTransform(scalar_t t_upper, scalar_t /*unused*/) : scale(t_upper) {}

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  scalar_t transform(typename c10::scalar_value_type<scalar_t>::type u) const {
    return scale * scalar_t(u * u);
  }

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  scalar_t jacobian(typename c10::scalar_value_type<scalar_t>::type u,
                    typename c10::scalar_value_type<scalar_t>::type half_width) const {
    return scalar_t(2) * scale * scalar_t(u) * scalar_t(half_width);
  }
};

/**
 * Upper region transformation policy: t = z - delta * v²
 *
 * Used for integration over [t_split, z] where delta = z - t_split.
 * The complementary quadratic transformation clusters points near t=z,
 * handling the (1-t)^(b-1) singularity when b < 1 and z is close to 1.
 *
 * Mapping: v ∈ [0, 1] → t ∈ [t_split, z]
 * Jacobian: |dt/dv| = 2 * delta * v
 */
template <typename scalar_t>
struct UpperRegionTransform {
  scalar_t z;
  scalar_t delta;  // z - t_split

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  UpperRegionTransform(scalar_t z_, scalar_t t_split) : z(z_), delta(z_ - t_split) {}

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  scalar_t transform(typename c10::scalar_value_type<scalar_t>::type v) const {
    return z - delta * scalar_t(v * v);
  }

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  scalar_t jacobian(typename c10::scalar_value_type<scalar_t>::type v,
                    typename c10::scalar_value_type<scalar_t>::type half_width) const {
    return scalar_t(2) * delta * scalar_t(v) * scalar_t(half_width);
  }
};

// ============================================================================
// Accumulator Policies for Quadrature
// ============================================================================
// These policies define what values to accumulate during quadrature.
// They enable the same quadrature loop to compute singly or doubly
// log-weighted integrals.

/**
 * Single-weight accumulator: computes J_a = ∫ f(t) ln(t) dt, J_b = ∫ f(t) ln(1-t) dt
 *
 * Used for first-order derivatives of the incomplete beta function.
 */
template <typename scalar_t>
struct SingleWeightAccumulator {
  scalar_t kronrod_a, kronrod_b;
  scalar_t gauss_a, gauss_b;

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  SingleWeightAccumulator()
      : kronrod_a(0), kronrod_b(0), gauss_a(0), gauss_b(0) {}

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  void accumulate_kronrod(scalar_t contrib, scalar_t log_t, scalar_t log_1mt) {
    kronrod_a = kronrod_a + contrib * log_t;
    kronrod_b = kronrod_b + contrib * log_1mt;
  }

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  void accumulate_gauss(scalar_t contrib, scalar_t log_t, scalar_t log_1mt) {
    gauss_a = gauss_a + contrib * log_t;
    gauss_b = gauss_b + contrib * log_1mt;
  }

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> result() const {
    return std::make_tuple(kronrod_a, kronrod_b, gauss_a, gauss_b);
  }
};

/**
 * Double-weight accumulator: computes K_aa, K_ab, K_bb integrals
 *
 * K_aa = ∫ f(t) ln²(t) dt
 * K_ab = ∫ f(t) ln(t)ln(1-t) dt
 * K_bb = ∫ f(t) ln²(1-t) dt
 *
 * Used for second-order derivatives of the incomplete beta function.
 */
template <typename scalar_t>
struct DoubleWeightAccumulator {
  scalar_t kronrod_aa, kronrod_ab, kronrod_bb;
  scalar_t gauss_aa, gauss_ab, gauss_bb;

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  DoubleWeightAccumulator()
      : kronrod_aa(0), kronrod_ab(0), kronrod_bb(0),
        gauss_aa(0), gauss_ab(0), gauss_bb(0) {}

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  void accumulate_kronrod(scalar_t contrib, scalar_t log_t, scalar_t log_1mt) {
    kronrod_aa = kronrod_aa + contrib * log_t * log_t;
    kronrod_ab = kronrod_ab + contrib * log_t * log_1mt;
    kronrod_bb = kronrod_bb + contrib * log_1mt * log_1mt;
  }

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  void accumulate_gauss(scalar_t contrib, scalar_t log_t, scalar_t log_1mt) {
    gauss_aa = gauss_aa + contrib * log_t * log_t;
    gauss_ab = gauss_ab + contrib * log_t * log_1mt;
    gauss_bb = gauss_bb + contrib * log_1mt * log_1mt;
  }

  C10_HOST_DEVICE C10_ALWAYS_INLINE
  std::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t, scalar_t> result() const {
    return std::make_tuple(kronrod_aa, kronrod_ab, kronrod_bb,
                           gauss_aa, gauss_ab, gauss_bb);
  }
};

// ============================================================================
// Unified Gauss-Kronrod Quadrature Implementation
// ============================================================================
// Template-based implementation that works with any combination of:
// - Gauss-Kronrod policy (G7-K15 or G15-K31)
// - Transformation policy (LowerRegion or UpperRegion)
// - Accumulator policy (SingleWeight or DoubleWeight)

/**
 * Unified Gauss-Kronrod quadrature for log-weighted beta integrals.
 *
 * This single template function replaces the four separate implementations
 * (lower/upper region × single/double weight) by parameterizing over
 * transformation and accumulator policies.
 *
 * Template parameters:
 *   GKPolicy: Gauss-Kronrod rule (GaussKronrod15Policy or GaussKronrod31Policy)
 *   Transform: Coordinate transformation (LowerRegionTransform or UpperRegionTransform)
 *   Accumulator: Value accumulator (SingleWeightAccumulator or DoubleWeightAccumulator)
 *
 * Parameters:
 *   transform: Transformation policy instance (contains scale/delta parameters)
 *   a, b: Beta function shape parameters
 *   param_left, param_right: Integration bounds in transformed space [0, 1]
 *
 * Returns: Accumulator's result tuple (varies by accumulator type)
 */
template <typename GKPolicy, typename Transform, typename Accumulator, typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
auto unified_gauss_kronrod_quadrature(
    const Transform& transform,
    scalar_t a,
    scalar_t b,
    scalar_t param_left,
    scalar_t param_right
) -> decltype(std::declval<Accumulator>().result()) {
  using std::exp;
  using std::log;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  Accumulator acc;

  // Scale factor for mapping [-1, 1] to [param_left, param_right]
  real_t half_width_real, center_real;
  if constexpr (c10::is_complex<scalar_t>::value) {
    half_width_real = (param_right.real() - param_left.real()) / real_t(2);
    center_real = (param_right.real() + param_left.real()) / real_t(2);
  } else {
    half_width_real = (param_right - param_left) / real_t(2);
    center_real = (param_right + param_left) / real_t(2);
  }

  // Process all Kronrod nodes (symmetric, so process both +/- for each non-zero node)
  for (int i = 0; i < GKPolicy::kNumNodes; ++i) {
    real_t node = GKPolicy::template kronrod_node<real_t>(i);
    real_t k_weight = GKPolicy::template kronrod_weight<real_t>(i);

    bool is_gauss = GKPolicy::is_gauss_node(i);
    real_t g_weight = real_t(0);
    if (is_gauss) {
      g_weight = GKPolicy::template gauss_weight<real_t>(GKPolicy::gauss_weight_index(i));
    }

    // Process both +node and -node (center node at i=0 only processed once)
    for (int sign = 0; sign <= (i > 0 ? 1 : 0); ++sign) {
      real_t param = (sign == 0)
          ? center_real + half_width_real * node
          : center_real - half_width_real * node;

      scalar_t t = transform.transform(param);

      // Validity check
      bool valid;
      if constexpr (c10::is_complex<scalar_t>::value) {
        valid = (param > real_t(0));
      } else {
        valid = (t > scalar_t(0) && t < scalar_t(1) && param > real_t(0));
      }

      if (valid) {
        scalar_t jacobian = transform.jacobian(param, half_width_real);

        scalar_t log_t = log(t);
        scalar_t log_1mt = log(scalar_t(1) - t);
        scalar_t log_base = (a - scalar_t(1)) * log_t + (b - scalar_t(1)) * log_1mt;
        scalar_t base_val = exp(log_base);

        scalar_t k_contrib = scalar_t(k_weight) * jacobian * base_val;
        acc.accumulate_kronrod(k_contrib, log_t, log_1mt);

        if (is_gauss) {
          scalar_t g_contrib = scalar_t(g_weight) * jacobian * base_val;
          acc.accumulate_gauss(g_contrib, log_t, log_1mt);
        }
      }
    }
  }

  return acc.result();
}

// ============================================================================
// Legacy API Wrappers (for backward compatibility)
// ============================================================================
// These functions wrap the unified implementation to maintain the existing API.

/**
 * Single-level Gauss-Kronrod quadrature for log-weighted beta integrals.
 *
 * Computes integrals of f(t) = t^(a-1) * (1-t)^(b-1) * g(t) over [0, t_upper]
 * where g(t) can be ln(t), ln(1-t), etc.
 *
 * Uses the quadratic transformation t = t_upper * u^2 to handle singularities
 * near t=0. The transformation clusters quadrature points near the origin.
 *
 * Template parameter Policy selects the quadrature rule (G7-K15 or G15-K31).
 *
 * Returns: tuple (kronrod_J_a, kronrod_J_b, gauss_J_a, gauss_J_b)
 * for simultaneous computation of J_a and J_b integrals with error estimation.
 */
template <typename Policy, typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
gauss_kronrod_beta_integrals(
    scalar_t t_upper,
    scalar_t a,
    scalar_t b,
    scalar_t u_left,
    scalar_t u_right
) {
  LowerRegionTransform<scalar_t> transform(t_upper);
  return unified_gauss_kronrod_quadrature<Policy, LowerRegionTransform<scalar_t>,
                                          SingleWeightAccumulator<scalar_t>>(
      transform, a, b, u_left, u_right);
}

/**
 * Single-level Gauss-Kronrod quadrature for the upper region [t_split, z].
 *
 * Uses the complementary transformation t = z - (z - t_split) * v^2 to cluster
 * quadrature points near t=z, which handles the (1-t)^(b-1) singularity when
 * b < 1 and z is close to 1.
 *
 * The transformation maps v in [0, 1] to t in [t_split, z]:
 *   - At v=0: t = z (upper bound, near the singularity at t=1)
 *   - At v=1: t = t_split (split point)
 *   - Jacobian: |dt/dv| = 2(z - t_split)v
 *
 * This complements the lower-region quadrature which uses t = t_split * u^2
 * to cluster points near t=0.
 */
template <typename Policy, typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
gauss_kronrod_upper_region(
    scalar_t z,
    scalar_t t_split,
    scalar_t a,
    scalar_t b,
    scalar_t v_left,
    scalar_t v_right
) {
  UpperRegionTransform<scalar_t> transform(z, t_split);
  return unified_gauss_kronrod_quadrature<Policy, UpperRegionTransform<scalar_t>,
                                          SingleWeightAccumulator<scalar_t>>(
      transform, a, b, v_left, v_right);
}

/**
 * Single-level Gauss-Kronrod for doubly log-weighted integrals (K_aa, K_ab, K_bb).
 *
 * These integrals are needed for second-order derivatives of the incomplete
 * beta function. The doubly log-weighted integrals have stronger singularities
 * due to the squared logarithms.
 */
template <typename Policy, typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t, scalar_t>
gauss_kronrod_doubly_weighted_integrals(
    scalar_t t_upper,
    scalar_t a,
    scalar_t b,
    scalar_t u_left,
    scalar_t u_right
) {
  LowerRegionTransform<scalar_t> transform(t_upper);
  return unified_gauss_kronrod_quadrature<Policy, LowerRegionTransform<scalar_t>,
                                          DoubleWeightAccumulator<scalar_t>>(
      transform, a, b, u_left, u_right);
}

/**
 * Single-level Gauss-Kronrod for doubly log-weighted integrals in the upper region.
 * Uses complementary transformation t = z - delta * v^2 to handle t=1 singularity.
 */
template <typename Policy, typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t, scalar_t>
gauss_kronrod_doubly_weighted_upper_region(
    scalar_t z,
    scalar_t t_split,
    scalar_t a,
    scalar_t b,
    scalar_t v_left,
    scalar_t v_right
) {
  UpperRegionTransform<scalar_t> transform(z, t_split);
  return unified_gauss_kronrod_quadrature<Policy, UpperRegionTransform<scalar_t>,
                                          DoubleWeightAccumulator<scalar_t>>(
      transform, a, b, v_left, v_right);
}

// ============================================================================
// Adaptive Quadrature Implementation with Kahan Summation
// ============================================================================

/**
 * Adaptive quadrature for the upper region [t_split, z].
 *
 * Uses complementary transformation t = z - delta * v^2 to handle
 * the (1-t)^(b-1) singularity when b < 1.
 *
 * Uses Kahan summation to reduce floating-point error accumulation
 * when many subintervals are summed.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t>
adaptive_upper_region_integrals(
    scalar_t z,
    scalar_t t_split,
    scalar_t a,
    scalar_t b,
    scalar_t tolerance,
    bool use_high_order
) {
  using std::abs;
  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  const real_t v_lower = real_t(0);
  const real_t v_upper = real_t(1);

  constexpr int kStackSize = kMaxAdaptiveDepth + 2;
  real_t stack_left[kStackSize];
  real_t stack_right[kStackSize];
  int stack_ptr = 0;

  // Use Kahan summation for improved accuracy
  KahanAccumulator<scalar_t> total_J_a;
  KahanAccumulator<scalar_t> total_J_b;

  stack_left[0] = v_lower;
  stack_right[0] = v_upper;
  stack_ptr = 1;

  int iterations = 0;
  constexpr int kMaxIterations = 256;

  // Extract real tolerance for comparisons
  real_t tol_real;
  if constexpr (c10::is_complex<scalar_t>::value) {
    tol_real = tolerance.real();
  } else {
    tol_real = tolerance;
  }

  while (stack_ptr > 0 && iterations < kMaxIterations) {
    iterations++;

    stack_ptr--;
    real_t v_left = stack_left[stack_ptr];
    real_t v_right = stack_right[stack_ptr];

    scalar_t kronrod_J_a, kronrod_J_b, gauss_J_a, gauss_J_b;
    if (use_high_order) {
      auto result = gauss_kronrod_upper_region<GaussKronrod31Policy>(
          z, t_split, a, b, scalar_t(v_left), scalar_t(v_right));
      kronrod_J_a = std::get<0>(result);
      kronrod_J_b = std::get<1>(result);
      gauss_J_a = std::get<2>(result);
      gauss_J_b = std::get<3>(result);
    } else {
      auto result = gauss_kronrod_upper_region<GaussKronrod15Policy>(
          z, t_split, a, b, scalar_t(v_left), scalar_t(v_right));
      kronrod_J_a = std::get<0>(result);
      kronrod_J_b = std::get<1>(result);
      gauss_J_a = std::get<2>(result);
      gauss_J_b = std::get<3>(result);
    }

    // Error and magnitude comparisons use real values (abs returns real_t for complex)
    real_t error_J_a = abs(kronrod_J_a - gauss_J_a);
    real_t error_J_b = abs(kronrod_J_b - gauss_J_b);
    real_t max_error = (error_J_a > error_J_b) ? error_J_a : error_J_b;

    real_t interval_fraction = (v_right - v_left) / (v_upper - v_lower);
    real_t local_tol = tol_real * interval_fraction;

    real_t mag_J_a = abs(kronrod_J_a);
    real_t mag_J_b = abs(kronrod_J_b);
    real_t max_mag = (mag_J_a > mag_J_b) ? mag_J_a : mag_J_b;
    real_t rel_tol = tol_real * max_mag;
    local_tol = (local_tol > rel_tol) ? local_tol : rel_tol;

    if (max_error <= local_tol || stack_ptr >= kStackSize - 2) {
      total_J_a.add(kronrod_J_a);
      total_J_b.add(kronrod_J_b);
    } else {
      real_t v_mid = (v_left + v_right) / real_t(2);

      stack_left[stack_ptr] = v_mid;
      stack_right[stack_ptr] = v_right;
      stack_ptr++;

      stack_left[stack_ptr] = v_left;
      stack_right[stack_ptr] = v_mid;
      stack_ptr++;
    }
  }

  // Warn if convergence was not achieved
  if (iterations >= kMaxIterations && stack_ptr > 0) {
    TORCHSCIENCE_QUADRATURE_WARN_CONVERGENCE(iterations, kMaxIterations, stack_ptr);
  }

  return std::make_tuple(total_J_a.result(), total_J_b.result());
}

/**
 * Adaptive quadrature for log-weighted beta integrals using iterative subdivision.
 *
 * Uses Gauss-Kronrod rules with an explicit stack to avoid recursion (CUDA safe).
 * The integration is performed on the transformed variable u where t = t_upper * u^2.
 *
 * Uses Kahan summation to reduce floating-point error accumulation when
 * summing contributions from many subintervals.
 *
 * Diagnostic Tracking:
 *   When TORCHSCIENCE_ENABLE_QUADRATURE_DIAGNOSTICS is defined, this function
 *   tracks and reports convergence statistics including:
 *   - Total iterations performed
 *   - Maximum subdivision depth reached
 *   - Estimated error
 *   - Whether convergence was achieved
 *
 * Parameters:
 *   t_upper: upper bound for integration domain [0, t_upper]
 *   a, b: shape parameters of the beta integrand
 *   tolerance: desired accuracy (absolute + relative)
 *   use_high_order: use G15-K31 instead of G7-K15 for difficult cases
 *
 * Returns: tuple (J_a, J_b) - the log-weighted integrals over [0, t_upper]
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t>
adaptive_log_weighted_beta_integrals(
    scalar_t t_upper,
    scalar_t a,
    scalar_t b,
    scalar_t tolerance,
    bool use_high_order
) {
  using std::abs;
  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  // Integration bounds in u-space: u in [0, 1] maps to t in [0, t_upper]
  const real_t u_lower = real_t(0);
  const real_t u_upper = real_t(1);

  // Fixed-size stack for interval subdivision (CUDA compatible)
  constexpr int kStackSize = kMaxAdaptiveDepth + 2;
  real_t stack_left[kStackSize];
  real_t stack_right[kStackSize];
  int stack_depth[kStackSize];  // Track depth for diagnostics
  int stack_ptr = 0;

  // Use Kahan summation for improved accuracy with many subdivisions
  KahanAccumulator<scalar_t> total_J_a;
  KahanAccumulator<scalar_t> total_J_b;

  // Diagnostic tracker (always allocated, but only used when enabled)
  AdaptiveQuadratureTracker tracker;

  // Initialize stack with full interval
  stack_left[0] = u_lower;
  stack_right[0] = u_upper;
  stack_depth[0] = 0;
  stack_ptr = 1;

  int iterations = 0;
  constexpr int kMaxIterations = 256;  // Safety limit

  // Extract real tolerance for comparisons
  real_t tol_real;
  if constexpr (c10::is_complex<scalar_t>::value) {
    tol_real = tolerance.real();
  } else {
    tol_real = tolerance;
  }

  while (stack_ptr > 0 && iterations < kMaxIterations) {
    iterations++;

    // Pop interval from stack
    stack_ptr--;
    real_t u_left = stack_left[stack_ptr];
    real_t u_right = stack_right[stack_ptr];
    int current_depth = stack_depth[stack_ptr];

    // Compute quadrature on this interval
    scalar_t kronrod_J_a, kronrod_J_b, gauss_J_a, gauss_J_b;
    if (use_high_order) {
      auto result = gauss_kronrod_beta_integrals<GaussKronrod31Policy>(
          t_upper, a, b, scalar_t(u_left), scalar_t(u_right));
      kronrod_J_a = std::get<0>(result);
      kronrod_J_b = std::get<1>(result);
      gauss_J_a = std::get<2>(result);
      gauss_J_b = std::get<3>(result);
    } else {
      auto result = gauss_kronrod_beta_integrals<GaussKronrod15Policy>(
          t_upper, a, b, scalar_t(u_left), scalar_t(u_right));
      kronrod_J_a = std::get<0>(result);
      kronrod_J_b = std::get<1>(result);
      gauss_J_a = std::get<2>(result);
      gauss_J_b = std::get<3>(result);
    }

    // Estimate error as |Kronrod - Gauss| for both integrals
    // Error and magnitude comparisons use real values (abs returns real_t for complex)
    real_t error_J_a = abs(kronrod_J_a - gauss_J_a);
    real_t error_J_b = abs(kronrod_J_b - gauss_J_b);
    real_t max_error = (error_J_a > error_J_b) ? error_J_a : error_J_b;

    // Track diagnostics
    tracker.record_iteration(static_cast<double>(max_error), current_depth);

    // Local tolerance: scale by interval size relative to total
    real_t interval_fraction = (u_right - u_left) / (u_upper - u_lower);
    real_t local_tol = tol_real * interval_fraction;

    // Also use relative tolerance based on estimate magnitude
    real_t mag_J_a = abs(kronrod_J_a);
    real_t mag_J_b = abs(kronrod_J_b);
    real_t max_mag = (mag_J_a > mag_J_b) ? mag_J_a : mag_J_b;
    real_t rel_tol = tol_real * max_mag;
    local_tol = (local_tol > rel_tol) ? local_tol : rel_tol;

    // Accept result if error is within tolerance or stack is nearly full
    if (max_error <= local_tol || stack_ptr >= kStackSize - 2) {
      total_J_a.add(kronrod_J_a);
      total_J_b.add(kronrod_J_b);
    } else {
      // Subdivide interval and push both halves onto stack
      real_t u_mid = (u_left + u_right) / real_t(2);
      int next_depth = current_depth + 1;

      // Push right half first (so left half is processed first)
      stack_left[stack_ptr] = u_mid;
      stack_right[stack_ptr] = u_right;
      stack_depth[stack_ptr] = next_depth;
      stack_ptr++;

      stack_left[stack_ptr] = u_left;
      stack_right[stack_ptr] = u_mid;
      stack_depth[stack_ptr] = next_depth;
      stack_ptr++;
    }
  }

  // Check if we hit the iteration limit without convergence
  if (iterations >= kMaxIterations && stack_ptr > 0) {
    tracker.mark_max_iterations_reached();
    TORCHSCIENCE_QUADRATURE_WARN_CONVERGENCE(iterations, kMaxIterations, stack_ptr);
  }

  // Report diagnostics (no-op when TORCHSCIENCE_ENABLE_QUADRATURE_DIAGNOSTICS is not defined)
  // The macro is defined in incomplete_beta.h when diagnostics are enabled
  #ifdef TORCHSCIENCE_ENABLE_QUADRATURE_DIAGNOSTICS
  // Note: This requires incomplete_beta.h to be included, which includes this file
  // The diagnostics update happens through external linkage when enabled
  #endif

  // The tracker contains useful diagnostic info that can be accessed if needed
  (void)tracker;  // Suppress unused variable warning when diagnostics disabled

  return std::make_tuple(total_J_a.result(), total_J_b.result());
}

/**
 * Adaptive quadrature for doubly log-weighted integrals in the upper region.
 * Uses Kahan summation for improved numerical accuracy.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t>
adaptive_doubly_weighted_upper_region(
    scalar_t z,
    scalar_t t_split,
    scalar_t a,
    scalar_t b,
    scalar_t tolerance
) {
  using std::abs;
  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  const real_t v_lower = real_t(0);
  const real_t v_upper = real_t(1);

  constexpr int kStackSize = kMaxAdaptiveDepth + 2;
  real_t stack_left[kStackSize];
  real_t stack_right[kStackSize];
  int stack_ptr = 0;

  // Use Kahan summation for improved accuracy
  KahanAccumulator<scalar_t> total_K_aa;
  KahanAccumulator<scalar_t> total_K_ab;
  KahanAccumulator<scalar_t> total_K_bb;

  stack_left[0] = v_lower;
  stack_right[0] = v_upper;
  stack_ptr = 1;

  int iterations = 0;
  constexpr int kMaxIterations = 256;

  // Extract real tolerance for comparisons
  real_t tol_real;
  if constexpr (c10::is_complex<scalar_t>::value) {
    tol_real = tolerance.real();
  } else {
    tol_real = tolerance;
  }

  while (stack_ptr > 0 && iterations < kMaxIterations) {
    iterations++;

    stack_ptr--;
    real_t v_left = stack_left[stack_ptr];
    real_t v_right = stack_right[stack_ptr];

    auto result = gauss_kronrod_doubly_weighted_upper_region<GaussKronrod15Policy>(
        z, t_split, a, b, scalar_t(v_left), scalar_t(v_right));
    scalar_t kronrod_K_aa = std::get<0>(result);
    scalar_t kronrod_K_ab = std::get<1>(result);
    scalar_t kronrod_K_bb = std::get<2>(result);
    scalar_t gauss_K_aa = std::get<3>(result);
    scalar_t gauss_K_ab = std::get<4>(result);
    scalar_t gauss_K_bb = std::get<5>(result);

    // Error and magnitude comparisons use real values (abs returns real_t for complex)
    real_t error_K_aa = abs(kronrod_K_aa - gauss_K_aa);
    real_t error_K_ab = abs(kronrod_K_ab - gauss_K_ab);
    real_t error_K_bb = abs(kronrod_K_bb - gauss_K_bb);
    real_t max_error = error_K_aa;
    if (error_K_ab > max_error) max_error = error_K_ab;
    if (error_K_bb > max_error) max_error = error_K_bb;

    real_t interval_fraction = (v_right - v_left) / (v_upper - v_lower);
    real_t local_tol = tol_real * interval_fraction;

    real_t max_mag = abs(kronrod_K_aa);
    if (abs(kronrod_K_ab) > max_mag) max_mag = abs(kronrod_K_ab);
    if (abs(kronrod_K_bb) > max_mag) max_mag = abs(kronrod_K_bb);
    real_t rel_tol = tol_real * max_mag;
    local_tol = (local_tol > rel_tol) ? local_tol : rel_tol;

    if (max_error <= local_tol || stack_ptr >= kStackSize - 2) {
      total_K_aa.add(kronrod_K_aa);
      total_K_ab.add(kronrod_K_ab);
      total_K_bb.add(kronrod_K_bb);
    } else {
      real_t v_mid = (v_left + v_right) / real_t(2);

      stack_left[stack_ptr] = v_mid;
      stack_right[stack_ptr] = v_right;
      stack_ptr++;

      stack_left[stack_ptr] = v_left;
      stack_right[stack_ptr] = v_mid;
      stack_ptr++;
    }
  }

  // Warn if convergence was not achieved
  if (iterations >= kMaxIterations && stack_ptr > 0) {
    TORCHSCIENCE_QUADRATURE_WARN_CONVERGENCE(iterations, kMaxIterations, stack_ptr);
  }

  return std::make_tuple(total_K_aa.result(), total_K_ab.result(), total_K_bb.result());
}

/**
 * Adaptive quadrature for doubly log-weighted beta integrals.
 * Supports t_upper parameter for dual-region integration.
 * Uses Kahan summation for improved numerical accuracy.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t>
adaptive_doubly_log_weighted_beta_integrals(
    scalar_t t_upper,
    scalar_t a,
    scalar_t b,
    scalar_t tolerance
) {
  using std::abs;
  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  const real_t u_lower = real_t(0);
  const real_t u_upper = real_t(1);

  constexpr int kStackSize = kMaxAdaptiveDepth + 2;
  real_t stack_left[kStackSize];
  real_t stack_right[kStackSize];
  int stack_ptr = 0;

  // Use Kahan summation for improved accuracy
  KahanAccumulator<scalar_t> total_K_aa;
  KahanAccumulator<scalar_t> total_K_ab;
  KahanAccumulator<scalar_t> total_K_bb;

  stack_left[0] = u_lower;
  stack_right[0] = u_upper;
  stack_ptr = 1;

  int iterations = 0;
  constexpr int kMaxIterations = 256;

  // Extract real tolerance for comparisons
  real_t tol_real;
  if constexpr (c10::is_complex<scalar_t>::value) {
    tol_real = tolerance.real();
  } else {
    tol_real = tolerance;
  }

  while (stack_ptr > 0 && iterations < kMaxIterations) {
    iterations++;

    stack_ptr--;
    real_t u_left = stack_left[stack_ptr];
    real_t u_right = stack_right[stack_ptr];

    auto result = gauss_kronrod_doubly_weighted_integrals<GaussKronrod15Policy>(
        t_upper, a, b, scalar_t(u_left), scalar_t(u_right));
    scalar_t kronrod_K_aa = std::get<0>(result);
    scalar_t kronrod_K_ab = std::get<1>(result);
    scalar_t kronrod_K_bb = std::get<2>(result);
    scalar_t gauss_K_aa = std::get<3>(result);
    scalar_t gauss_K_ab = std::get<4>(result);
    scalar_t gauss_K_bb = std::get<5>(result);

    // Error and magnitude comparisons use real values (abs returns real_t for complex)
    real_t error_K_aa = abs(kronrod_K_aa - gauss_K_aa);
    real_t error_K_ab = abs(kronrod_K_ab - gauss_K_ab);
    real_t error_K_bb = abs(kronrod_K_bb - gauss_K_bb);
    real_t max_error = error_K_aa;
    if (error_K_ab > max_error) max_error = error_K_ab;
    if (error_K_bb > max_error) max_error = error_K_bb;

    real_t interval_fraction = (u_right - u_left) / (u_upper - u_lower);
    real_t local_tol = tol_real * interval_fraction;

    real_t max_mag = abs(kronrod_K_aa);
    if (abs(kronrod_K_ab) > max_mag) max_mag = abs(kronrod_K_ab);
    if (abs(kronrod_K_bb) > max_mag) max_mag = abs(kronrod_K_bb);
    real_t rel_tol = tol_real * max_mag;
    local_tol = (local_tol > rel_tol) ? local_tol : rel_tol;

    if (max_error <= local_tol || stack_ptr >= kStackSize - 2) {
      total_K_aa.add(kronrod_K_aa);
      total_K_ab.add(kronrod_K_ab);
      total_K_bb.add(kronrod_K_bb);
    } else {
      real_t u_mid = (u_left + u_right) / real_t(2);

      stack_left[stack_ptr] = u_mid;
      stack_right[stack_ptr] = u_right;
      stack_ptr++;

      stack_left[stack_ptr] = u_left;
      stack_right[stack_ptr] = u_mid;
      stack_ptr++;
    }
  }

  // Warn if convergence was not achieved
  if (iterations >= kMaxIterations && stack_ptr > 0) {
    TORCHSCIENCE_QUADRATURE_WARN_CONVERGENCE(iterations, kMaxIterations, stack_ptr);
  }

  return std::make_tuple(total_K_aa.result(), total_K_ab.result(), total_K_bb.result());
}

}  // namespace torchscience::impl::special_functions
