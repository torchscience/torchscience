#pragma once

/*
 * Double backward pass for Regularized Incomplete Beta Function I_z(a, b)
 *
 * This file contains the second-order derivative (double backward) implementations
 * for the incomplete beta function. See incomplete_beta.h for the forward
 * implementation and incomplete_beta_backward.h for the first backward.
 */

#include "incomplete_beta_backward.h"
#include "trigamma.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Fused double-backward implementation (second-order derivatives)
// ============================================================================

/**
 * Fused double-backward computation with fully analytical second derivatives.
 *
 * All second-order derivatives are computed analytically using:
 * - Trigamma functions for d^2I/da^2, d^2I/db^2, d^2I/dadb
 * - Doubly log-weighted integrals K_aa, K_ab, K_bb
 *
 * Analytical formulas:
 *   d^2I/dz^2 = dI/dz * [(a-1)/z - (b-1)/(1-z)]
 *   d^2I/dzda = dI/dz * [log(z) - psi(a) + psi(a+b)]
 *   d^2I/dzdb = dI/dz * [log(1-z) - psi(b) + psi(a+b)]
 *
 *   d^2I/da^2 = K_aa/B - 2*(J_a/B)*(psi(a) - psi(a+b))
 *           + I_z*(psi(a) - psi(a+b))^2 - I_z*(psi'(a) - psi'(a+b))
 *
 *   d^2I/db^2 = K_bb/B - 2*(J_b/B)*(psi(b) - psi(a+b))
 *           + I_z*(psi(b) - psi(a+b))^2 - I_z*(psi'(b) - psi'(a+b))
 *
 *   d^2I/dadb = K_ab/B - (J_a/B)*(psi(b) - psi(a+b)) - (J_b/B)*(psi(a) - psi(a+b))
 *            + I_z*(psi(a) - psi(a+b))*(psi(b) - psi(a+b)) + I_z*psi'(a+b)
 *
 * where B = B(a,b), psi = digamma, psi' = trigamma, and K_xx are doubly
 * log-weighted integrals computed via adaptive Gauss-Kronrod quadrature.
 */

// Forward declaration of double-backward for extended region
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T, T>
incomplete_beta_backward_backward(
    T gradient_gradient_z, T gradient_gradient_a, T gradient_gradient_b,
    T gradient_output, T z, T a, T b,
    bool has_gradient_gradient_z, bool has_gradient_gradient_a, bool has_gradient_gradient_b
);

/**
 * Double backward for analytic continuation region (|z| >= 1).
 *
 * Region B (|1-z| < 1): Use chain rule through symmetry relation
 * Region C (|1-z| >= 1): Use finite differences (analytical formulas are very complex)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T, T>
incomplete_beta_backward_backward_extended(
    T gradient_gradient_z,
    T gradient_gradient_a,
    T gradient_gradient_b,
    T gradient_output,
    T z,
    T a,
    T b,
    bool has_gradient_gradient_z,
    bool has_gradient_gradient_a,
    bool has_gradient_gradient_b
) {
  using std::abs;
  using std::conj;

  using real_t = typename c10::scalar_value_type<T>::type;

  T gradient_gradient_output = T(0);
  T gradient_z = T(0);
  T gradient_a = T(0);
  T gradient_b = T(0);

  if (!has_gradient_gradient_z && !has_gradient_gradient_a && !has_gradient_gradient_b) {
    return std::make_tuple(gradient_gradient_output, gradient_z, gradient_a, gradient_b);
  }

  // Compute |1-z|
  T one_minus_z = T(1) - z;
  real_t one_minus_z_mag;
  if constexpr (c10::is_complex<T>::value) {
    one_minus_z_mag = abs(one_minus_z);
  } else {
    one_minus_z_mag = abs(one_minus_z);
  }

  // Region B: |z| >= 1 but |1-z| < 1 - use symmetry relation
  if (one_minus_z_mag < real_t(1)) {
    // I_z(a,b) = 1 - I_{1-z}(b,a)
    // The gradient transformations follow the same pattern as first backward:
    // gradient_z transforms to gradient_w at (1-z, b, a)
    // gradient_a transforms from the third argument
    // gradient_b transforms from the second argument

    // Call double-backward at transformed point
    auto [
      gg_out_t,
      grad_w,
      grad_first,
      grad_second
    ] = incomplete_beta_backward_backward(
        gradient_gradient_z, gradient_gradient_b, gradient_gradient_a,  // Note: swap gradient_gradient_a and gradient_gradient_b since args are swapped
        gradient_output, one_minus_z, b, a,
        has_gradient_gradient_z, has_gradient_gradient_b, has_gradient_gradient_a  // Note: swap has_gradient_gradient_a and has_gradient_gradient_b
    );

    // Transform gradients back
    gradient_gradient_output = gg_out_t;
    gradient_z = -grad_w;  // w = 1-z transformation, dw/dz = -1
    gradient_a = -grad_second;  // a is third arg in I(w,b,a)
    gradient_b = -grad_first;   // b is second arg in I(w,b,a)

    return std::make_tuple(gradient_gradient_output, gradient_z, gradient_a, gradient_b);
  }

  // Region C: Both |z| > 1 and |1-z| >= 1
  // Use finite differences for second derivatives (analytical formulas are very complex)
  real_t delta = real_t(1e-6);

  // Get first derivatives via backward
  auto [
    grad_z,
    grad_a,
    grad_b
  ] = incomplete_beta_backward_extended(T(1), z, a, b);

  if (has_gradient_gradient_z) {
    gradient_gradient_output = gradient_gradient_output + gradient_gradient_z * grad_z;

    auto [
      grad_z_plus,
      grad_a_plus,
      grad_b_plus
    ] = incomplete_beta_backward_extended(T(1), z + T(delta), a, b);

    if constexpr (c10::is_complex<T>::value) {
      gradient_z = gradient_z + conj(gradient_gradient_z) * gradient_output * conj((grad_z_plus - grad_z) / T(delta));
      gradient_a = gradient_a + conj(gradient_gradient_z) * gradient_output * conj((grad_a_plus - grad_a) / T(delta));
      gradient_b = gradient_b + conj(gradient_gradient_z) * gradient_output * conj((grad_b_plus - grad_b) / T(delta));
    } else {
      gradient_z = gradient_z + gradient_gradient_z * gradient_output * ((grad_z_plus - grad_z) / T(delta));
      gradient_a = gradient_a + gradient_gradient_z * gradient_output * ((grad_a_plus - grad_a) / T(delta));
      gradient_b = gradient_b + gradient_gradient_z * gradient_output * ((grad_b_plus - grad_b) / T(delta));
    }
  }

  if (has_gradient_gradient_a) {
    gradient_gradient_output = gradient_gradient_output + gradient_gradient_a * grad_a;

    // d^2I/da^2 and d^2I/dadb via finite difference
    auto [grad_z_plus_a, grad_a_plus_a, grad_b_plus_a] = incomplete_beta_backward_extended(
        T(1), z, a + T(delta), b);
    T d2Idadz = (grad_z_plus_a - grad_z) / T(delta);
    T d2Ida2 = (grad_a_plus_a - grad_a) / T(delta);
    T d2Idadb = (grad_b_plus_a - grad_b) / T(delta);

    if constexpr (c10::is_complex<T>::value) {
      gradient_z = gradient_z + conj(gradient_gradient_a) * gradient_output * conj(d2Idadz);
      gradient_a = gradient_a + conj(gradient_gradient_a) * gradient_output * conj(d2Ida2);
      gradient_b = gradient_b + conj(gradient_gradient_a) * gradient_output * conj(d2Idadb);
    } else {
      gradient_z = gradient_z + gradient_gradient_a * gradient_output * d2Idadz;
      gradient_a = gradient_a + gradient_gradient_a * gradient_output * d2Ida2;
      gradient_b = gradient_b + gradient_gradient_a * gradient_output * d2Idadb;
    }
  }

  if (has_gradient_gradient_b) {
    gradient_gradient_output = gradient_gradient_output + gradient_gradient_b * grad_b;

    auto [
      grad_z_plus_b,
      grad_a_plus_b,
      grad_b_plus_b
    ] = incomplete_beta_backward_extended( T(1), z, a, b + T(delta));

    T d2Idbdz = (grad_z_plus_b - grad_z) / T(delta);
    T d2Idbda = (grad_a_plus_b - grad_a) / T(delta);
    T d2Idb2 = (grad_b_plus_b - grad_b) / T(delta);

    if constexpr (c10::is_complex<T>::value) {
      gradient_z = gradient_z + conj(gradient_gradient_b) * gradient_output * conj(d2Idbdz);
      gradient_a = gradient_a + conj(gradient_gradient_b) * gradient_output * conj(d2Idbda);
      gradient_b = gradient_b + conj(gradient_gradient_b) * gradient_output * conj(d2Idb2);
    } else {
      gradient_z = gradient_z + gradient_gradient_b * gradient_output * d2Idbdz;
      gradient_a = gradient_a + gradient_gradient_b * gradient_output * d2Idbda;
      gradient_b = gradient_b + gradient_gradient_b * gradient_output * d2Idb2;
    }
  }

  return std::make_tuple(
    gradient_gradient_output,
    gradient_z,
    gradient_a,
    gradient_b
    );
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T, T> incomplete_beta_backward_backward(
    T gradient_gradient_z,
    T gradient_gradient_a,
    T gradient_gradient_b,
    T gradient_output,
    T z,
    T a,
    T b,
    bool has_gradient_gradient_z,
    bool has_gradient_gradient_a,
    bool has_gradient_gradient_b
) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::conj;

  using R = typename c10::scalar_value_type<T>::type;

  T gradient_gradient_output = T(0);
  T gradient_z = T(0);
  T gradient_a = T(0);
  T gradient_b = T(0);

  if constexpr (c10::is_complex<T>::value) {
    // Complex double-backward using Wirtinger derivative convention.
    //
    // For PyTorch's complex autograd, the first backward computes:
    //   B_x = grad_output * conj(∂f/∂x)
    //
    // where B_x is holomorphic in grad_output but anti-holomorphic in x
    // (since conj(∂f/∂x) is anti-holomorphic for holomorphic f).
    //
    // For the double backward with gg_x = ∂L/∂(B_x)*:
    //
    // 1. Contribution to ∂L/∂(grad_output)*:
    //    B_x is holomorphic in grad_output, so ∂B_x/∂(grad_output)* = 0
    //    but ∂(B_x)*/∂(grad_output)* = ∂f/∂x (not conjugated)
    //    Thus: ∂L/∂(grad_output)* = gg_x * ∂f/∂x
    //
    // 2. Contribution to ∂L/∂x*:
    //    B_x is anti-holomorphic in x, so:
    //    - ∂B_x/∂x* = grad_output * conj(∂²f/∂x²)
    //    - ∂(B_x)*/∂x* = 0
    //    And ∂L/∂B_x = conj(gg_x)
    //    Thus: ∂L/∂x* = conj(gg_x) * grad_output * conj(∂²f/∂x²)

    const R tol = R(1e-10);

    if (!has_gradient_gradient_z && !has_gradient_gradient_a && !has_gradient_gradient_b) {
      return std::make_tuple(gradient_gradient_output, gradient_z, gradient_a, gradient_b);
    }

    if (abs(z) < tol || abs(z - T(1)) < tol) {
      return std::make_tuple(
        gradient_gradient_output,
        gradient_z,
        gradient_a,
        gradient_b
      );
    }

    if (abs(z) >= R(1)) {
      return incomplete_beta_backward_backward_extended(
        gradient_gradient_z,
        gradient_gradient_a,
        gradient_gradient_b,
        gradient_output,
        z,
        a,
        b,
        has_gradient_gradient_z,
        has_gradient_gradient_a,
        has_gradient_gradient_b
      );
    }

    // Invalid parameters
    if (a.real() <= R(0) || b.real() <= R(0)) {
      return std::make_tuple(gradient_gradient_output, gradient_z, gradient_a, gradient_b);
    }

    DiGammaCache<T> psi_cache(a, b);

    auto [
      J_a,
      J_b
    ] = log_weighted_beta_integrals(z, a, b);

    if (has_gradient_gradient_z) {
      gradient_gradient_output = gradient_gradient_output + gradient_gradient_z * exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b));

      gradient_z = gradient_z + conj(gradient_gradient_z) * gradient_output * conj(exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b)) * ((a - T(1)) / z - (b - T(1)) / (T(1) - z)));
      gradient_a = gradient_a + conj(gradient_gradient_z) * gradient_output * conj(exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b)) * (log(z) - psi_cache.psi_a_minus_ab));
      gradient_b = gradient_b + conj(gradient_gradient_z) * gradient_output * conj(exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b)) * (log(T(1) - z) - psi_cache.psi_b_minus_ab));
    }

    if (has_gradient_gradient_a || has_gradient_gradient_b) {
      auto [
        K_aa,
        K_ab,
        K_bb
      ] = doubly_log_weighted_beta_integrals(z, a, b);

      T K_aa_over_B = K_aa * exp(-log_beta(a, b));
      T K_ab_over_B = K_ab * exp(-log_beta(a, b));
      T K_bb_over_B = K_bb * exp(-log_beta(a, b));

      if (has_gradient_gradient_a) {
        gradient_gradient_output = gradient_gradient_output + gradient_gradient_a * (J_a * exp(-log_beta(a, b)) - incomplete_beta(z, a, b) * psi_cache.psi_a_minus_ab);

        gradient_z = gradient_z + conj(gradient_gradient_a) * gradient_output * conj(exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b)) * (log(z) - psi_cache.psi_a_minus_ab));
        gradient_a = gradient_a + conj(gradient_gradient_a) * gradient_output * conj(K_aa_over_B - T(2) * (J_a * exp(-log_beta(a, b))) * psi_cache.psi_a_minus_ab + incomplete_beta(z, a, b) * (psi_cache.psi_a_minus_ab * psi_cache.psi_a_minus_ab) - incomplete_beta(z, a, b) * (trigamma(a) - trigamma(a + b)));
        gradient_b = gradient_b + conj(gradient_gradient_a) * gradient_output * conj(K_ab_over_B - J_a * exp(-log_beta(a, b)) * psi_cache.psi_b_minus_ab - J_b * exp(-log_beta(a, b)) * psi_cache.psi_a_minus_ab + incomplete_beta(z, a, b) * psi_cache.psi_a_minus_ab * psi_cache.psi_b_minus_ab + incomplete_beta(z, a, b) * trigamma(a + b));
      }

      if (has_gradient_gradient_b) {
        gradient_gradient_output = gradient_gradient_output + gradient_gradient_b * (J_b * exp(-log_beta(a, b)) - incomplete_beta(z, a, b) * psi_cache.psi_b_minus_ab);

        gradient_z = gradient_z + conj(gradient_gradient_b) * gradient_output * conj(exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b)) * (log(T(1) - z) - psi_cache.psi_b_minus_ab));
        gradient_a = gradient_a + conj(gradient_gradient_b) * gradient_output * conj(K_ab_over_B - J_a * exp(-log_beta(a, b)) * psi_cache.psi_b_minus_ab - J_b * exp(-log_beta(a, b)) * psi_cache.psi_a_minus_ab + incomplete_beta(z, a, b) * psi_cache.psi_a_minus_ab * psi_cache.psi_b_minus_ab + incomplete_beta(z, a, b) * trigamma(a + b));
        gradient_b = gradient_b + conj(gradient_gradient_b) * gradient_output * conj(K_bb_over_B - T(2) * (J_b * exp(-log_beta(a, b))) * psi_cache.psi_b_minus_ab + incomplete_beta(z, a, b) * (psi_cache.psi_b_minus_ab * psi_cache.psi_b_minus_ab) - incomplete_beta(z, a, b) * (trigamma(b) - trigamma(a + b)));
      }
    }

    return std::make_tuple(gradient_gradient_output, gradient_z, gradient_a, gradient_b);
  } else {
    if (!has_gradient_gradient_z && !has_gradient_gradient_a && !has_gradient_gradient_b) {
      return std::make_tuple(
        gradient_gradient_output,
        gradient_z,
        gradient_a,
        gradient_b
      );
    }

    if (z <= T(0) || z >= T(1) || a <= T(0) || b <= T(0)) {
      return std::make_tuple(
        gradient_gradient_output,
        gradient_z,
        gradient_a,
        gradient_b
      );
    }

    DiGammaCache<T> psi_cache(a, b);

    auto [
      J_a,
      J_b
    ] = log_weighted_beta_integrals(z, a, b);

    if (has_gradient_gradient_z) {
      gradient_gradient_output = gradient_gradient_output + gradient_gradient_z * exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b));

      gradient_z = gradient_z + gradient_gradient_z * gradient_output * (exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b)) * ((a - T(1)) / z - (b - T(1)) / (T(1) - z)));
      gradient_a = gradient_a + gradient_gradient_z * gradient_output * (exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b)) * (log(z) - psi_cache.psi_a_minus_ab));
      gradient_b = gradient_b + gradient_gradient_z * gradient_output * (exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b)) * (log(T(1) - z) - psi_cache.psi_b_minus_ab));
    }

    if (has_gradient_gradient_a || has_gradient_gradient_b) {
      auto [
        K_aa,
        K_ab,
        K_bb
      ] = doubly_log_weighted_beta_integrals(z, a, b);

      if (has_gradient_gradient_a) {
        gradient_gradient_output = gradient_gradient_output + gradient_gradient_a * (J_a * exp(-log_beta(a, b)) - incomplete_beta(z, a, b) * psi_cache.psi_a_minus_ab);

        gradient_z = gradient_z + gradient_gradient_a * gradient_output * (exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b)) * (log(z) - psi_cache.psi_a_minus_ab));
        gradient_a = gradient_a + gradient_gradient_a * gradient_output * (K_aa * exp(-log_beta(a, b)) - T(2) * (J_a * exp(-log_beta(a, b))) * psi_cache.psi_a_minus_ab + incomplete_beta(z, a, b) * (psi_cache.psi_a_minus_ab * psi_cache.psi_a_minus_ab) - incomplete_beta(z, a, b) * (trigamma(a) - trigamma(a + b)));
        gradient_b = gradient_b + gradient_gradient_a * gradient_output * (K_ab * exp(-log_beta(a, b)) - J_a * exp(-log_beta(a, b)) * psi_cache.psi_b_minus_ab - J_b * exp(-log_beta(a, b)) * psi_cache.psi_a_minus_ab + incomplete_beta(z, a, b) * psi_cache.psi_a_minus_ab * psi_cache.psi_b_minus_ab + incomplete_beta(z, a, b) * trigamma(a + b));
      }

      if (has_gradient_gradient_b) {
        gradient_gradient_output = gradient_gradient_output + gradient_gradient_b * (J_b * exp(-log_beta(a, b)) - incomplete_beta(z, a, b) * psi_cache.psi_b_minus_ab);

        gradient_z = gradient_z + gradient_gradient_b * gradient_output * (exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b)) * (log(T(1) - z) - psi_cache.psi_b_minus_ab));
        gradient_a = gradient_a + gradient_gradient_b * gradient_output * (K_ab * exp(-log_beta(a, b)) - J_a * exp(-log_beta(a, b)) * psi_cache.psi_b_minus_ab - J_b * exp(-log_beta(a, b)) * psi_cache.psi_a_minus_ab + incomplete_beta(z, a, b) * psi_cache.psi_a_minus_ab * psi_cache.psi_b_minus_ab + incomplete_beta(z, a, b) * trigamma(a + b));
        gradient_b = gradient_b + gradient_gradient_b * gradient_output * (K_bb * exp(-log_beta(a, b)) - T(2) * (J_b * exp(-log_beta(a, b))) * psi_cache.psi_b_minus_ab + incomplete_beta(z, a, b) * (psi_cache.psi_b_minus_ab * psi_cache.psi_b_minus_ab) - incomplete_beta(z, a, b) * (trigamma(b) - trigamma(a + b)));
      }
    }

    return std::make_tuple(
      gradient_gradient_output,
      gradient_z,
      gradient_a,
      gradient_b
    );
  }
}

}  // namespace torchscience::impl::special_functions
