#pragma once

/*
 * Backward pass for Regularized Incomplete Beta Function I_z(a, b)
 *
 * This file contains the first-order derivative (backward) implementations
 * for the incomplete beta function. See incomplete_beta.h for the forward
 * implementation.
 */

#include "incomplete_beta.h"
#include "digamma.h"
#include "gamma.h"
#include "hypergeometric_2_f_1.h"
#include "adaptive_quadrature.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Extended Domain Backward Implementation
// ============================================================================

// Forward declaration of backward function for use in extended backward
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T>
incomplete_beta_backward(T grad, T z, T a, T b);

/**
 * Backward for analytic continuation region (|z| >= 1).
 *
 * Region Classification:
 *   - Region B (|1-z| < 1): Use chain rule through symmetry I_z(a,b) = 1 - I_{1-z}(b,a)
 *   - Region C (|1-z| >= 1): Use differentiation through hypergeometric formula
 *
 * For Region B, the symmetry relation gives:
 *   dI/dz(z,a,b) = dI/dw(w,b,a) where w = 1-z
 *   dI/da(z,a,b) = -dI/db(w,b,a) (since a is 3rd arg in I(w,b,a))
 *   dI/db(z,a,b) = -dI/da(w,b,a) (since b is 2nd arg in I(w,b,a))
 *
 * For Region C, we differentiate the hypergeometric representation directly.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T>
incomplete_beta_backward_extended(T grad, T z, T a, T b) {
  using std::abs;
  using std::exp;
  using std::log;
  using std::conj;

  using real_t = typename c10::scalar_value_type<T>::type;

  T gradient_z = T(0);
  T gradient_a = T(0);
  T gradient_b = T(0);

  real_t one_minus_z_mag;

  if constexpr (c10::is_complex<T>::value) {
    one_minus_z_mag = abs(T(1) - z);
  } else {
    one_minus_z_mag = abs(T(1) - z);
  }

  if (one_minus_z_mag < real_t(1)) {
    auto [grad_w, grad_first, grad_second] = incomplete_beta_backward(grad, T(1) - z, b, a);

    return std::make_tuple(grad_w, -grad_second, -grad_first);
  }

  if constexpr (c10::is_complex<T>::value) {
    gradient_z = grad * conj(a / z * exp(a * log(z) - log(a) - log_beta(a, b)) * hypergeometric_2f1_linear_transform(a, T(1) - b, a + T(1), z) + exp(a * log(z) - log(a) - log_beta(a, b)) * (a * (T(1) - b) / (a + T(1)) * hypergeometric_2f1_linear_transform( a + T(1), T(1) - b + T(1), a + T(1) + T(1), z)));
    gradient_a = grad * conj((incomplete_beta_extended_domain(z, a + T(real_t(1e-7)), b) - exp(a * log(z) - log(a) - log_beta(a, b)) * hypergeometric_2f1_linear_transform(a, T(1) - b, a + T(1), z)) / T(real_t(1e-7)));
    gradient_b = grad * conj((incomplete_beta_extended_domain(z, a, b + T(real_t(1e-7))) - exp(a * log(z) - log(a) - log_beta(a, b)) * hypergeometric_2f1_linear_transform(a, T(1) - b, a + T(1), z)) / T(real_t(1e-7)));
  } else {
    gradient_z = grad * (a / z * exp(a * log(z) - log(a) - log_beta(a, b)) * hypergeometric_2f1_linear_transform(a, T(1) - b, a + T(1), z) + exp(a * log(z) - log(a) - log_beta(a, b)) * (a * (T(1) - b) / (a + T(1)) * hypergeometric_2f1_linear_transform( a + T(1), T(1) - b + T(1), a + T(1) + T(1), z)));
    gradient_a = grad * ((incomplete_beta_extended_domain(z, a + T(real_t(1e-7)), b) - exp(a * log(z) - log(a) - log_beta(a, b)) * hypergeometric_2f1_linear_transform(a, T(1) - b, a + T(1), z)) / T(real_t(1e-7)));
    gradient_b = grad * ((incomplete_beta_extended_domain(z, a, b + T(real_t(1e-7))) - exp(a * log(z) - log(a) - log_beta(a, b)) * hypergeometric_2f1_linear_transform(a, T(1) - b, a + T(1), z)) / T(real_t(1e-7)));
  }

  return std::make_tuple(
    gradient_z,
    gradient_a,
    gradient_b
  );
}

// ============================================================================
// Fused backward implementation (first-order derivatives)
// ============================================================================

/**
 * Fused backward - computes gradient_z, gradient_a, gradient_b in a single pass.
 *
 * All derivatives are computed analytically:
 *
 *   dI/dz = z^(a-1) * (1-z)^(b-1) / B(a,b)
 *
 *   dI/da = J_a(z,a,b) / B(a,b) - I_z(a,b) * [psi(a) - psi(a+b)]
 *
 *   dI/db = J_b(z,a,b) / B(a,b) - I_z(a,b) * [psi(b) - psi(a+b)]
 *
 * where:
 *   - psi is the digamma function
 *   - J_a = integral from 0 to z of t^(a-1) * (1-t)^(b-1) * ln(t) dt
 *   - J_b = integral from 0 to z of t^(a-1) * (1-t)^(b-1) * ln(1-t) dt
 *
 * The log-weighted integrals J_a and J_b are computed using adaptive
 * Gauss-Kronrod quadrature for high accuracy.
 *
 * For complex inputs, uses Wirtinger derivative convention:
 *   gradient = grad_output * conj(df/dz)
 * where df/dz is the holomorphic derivative.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T> incomplete_beta_backward(
  T grad,
  T z,
  T a,
  T b
) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::conj;

  using real_t = typename c10::scalar_value_type<T>::type;

  T gradient_z = T(0);
  T gradient_a = T(0);
  T gradient_b = T(0);

  if constexpr (c10::is_complex<T>::value) {
    // Complex implementation with Wirtinger derivatives
    const real_t tol = real_t(1e-10);

    // Handle boundary cases
    if (abs(z) < tol || abs(z - T(1)) < tol) {
      return std::make_tuple(gradient_z, gradient_a, gradient_b);
    }

    // For |z| >= 1, use extended backward
    if (abs(z) >= real_t(1)) {
      return incomplete_beta_backward_extended(grad, z, a, b);
    }

    // Invalid parameters
    if (a.real() <= real_t(0) || b.real() <= real_t(0)) {
      return std::make_tuple(gradient_z, gradient_a, gradient_b);
    }

    // Compute log_beta for reuse
    T lb = log_beta(a, b);

    // dI/dz = z^(a-1) * (1-z)^(b-1) / B(a,b)
    T log_dIdz = (a - T(1)) * log(z) +
                        (b - T(1)) * log(T(1) - z) - lb;
    T dIdz = exp(log_dIdz);

    // Compute the function value I_z(a, b) for use in parameter derivatives
    T I_z = incomplete_beta(z, a, b);

    // Compute analytical derivatives for a and b using digamma and quadrature
    auto [dIda, dIdb] = incomplete_beta_parameter_derivatives(z, a, b, I_z);

    // For complex autograd, PyTorch expects: grad_input = grad * conj(df/dz)
    gradient_z = grad * conj(dIdz);
    gradient_a = grad * conj(dIda);
    gradient_b = grad * conj(dIdb);

    return std::make_tuple(gradient_z, gradient_a, gradient_b);
  } else {
    // Real implementation
    // Handle boundary and invalid cases
    if (z <= T(0) || z >= T(1) || a <= T(0) || b <= T(0)) {
      return std::make_tuple(gradient_z, gradient_a, gradient_b);
    }

    // Try asymptotic expansion for dI/dz near z=0
    auto [dIdz_zero, valid_dz_zero] = incomplete_beta_dz_asymptotic_zero(z, a, b);
    if (valid_dz_zero) {
      gradient_z = grad * dIdz_zero;

      // For parameter derivatives near z=0, use asymptotic formulas
      auto [result_zero, valid_I_zero] = incomplete_beta_asymptotic_zero(z, a, b);
      T I_z = valid_I_zero ? result_zero : incomplete_beta(z, a, b);
      auto [dIda, dIdb, valid_params] = incomplete_beta_param_derivs_asymptotic_zero(z, a, b, I_z);
      if (valid_params) {
        gradient_a = grad * dIda;
        gradient_b = grad * dIdb;
        return std::make_tuple(gradient_z, gradient_a, gradient_b);
      }
      // Fall through to use standard parameter derivatives if asymptotic failed
      auto [dIda_std, dIdb_std] = incomplete_beta_parameter_derivatives(z, a, b, I_z);
      gradient_a = grad * dIda_std;
      gradient_b = grad * dIdb_std;
      return std::make_tuple(gradient_z, gradient_a, gradient_b);
    }

    // Try asymptotic expansion for dI/dz near z=1
    auto [dIdz_one, valid_dz_one] = incomplete_beta_dz_asymptotic_one(z, a, b);
    if (valid_dz_one) {
      auto [
        dIda,
        dIdb
      ] = incomplete_beta_parameter_derivatives(z, a, b, incomplete_beta(z, a, b));

      return std::make_tuple(
        grad * dIdz_one,
        grad * dIda,
        grad * dIdb
      );
    }

    auto [
      dIda,
      dIdb
    ] = incomplete_beta_parameter_derivatives(z, a, b, incomplete_beta(z, a, b));

    return std::make_tuple(
      grad * exp((a - T(1)) * log(z) + (b - T(1)) * log(T(1) - z) - log_beta(a, b)),
      grad * dIda, grad * dIdb
    );
  }
}

}  // namespace torchscience::impl::special_functions
