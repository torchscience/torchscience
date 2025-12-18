#pragma once

/*
 * Gauss Hypergeometric Function 2F1(a, b; c; z)
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    The Gauss hypergeometric function is defined by the series:
 *
 *    2F1(a, b; c; z) = sum_{n=0}^{inf} (a)_n * (b)_n / ((c)_n * n!) * z^n
 *
 *    where (x)_n = x*(x+1)*...*(x+n-1) is the Pochhammer symbol.
 *
 * 2. CONVERGENCE:
 *    - Absolutely convergent for |z| < 1
 *    - Conditionally convergent for |z| = 1 if Re(c - a - b) > 0
 *    - Divergent for |z| > 1 (requires analytic continuation)
 *
 * 3. ALGORITHM SELECTION:
 *    The implementation uses three different algorithms based on |z|:
 *
 *    a) Direct series (|z| < 0.5 or |1-z| >= |z|):
 *       Uses the defining series expansion.
 *
 *    b) 1-z transformation (|z| < 1 and |1-z| < |z|, DLMF 15.8.4):
 *       2F1(a,b;c;z) = H1 * 2F1(a, b; a+b-c+1; 1-z)
 *                    + H2 * (1-z)^{c-a-b} * 2F1(c-a, c-b; c-a-b+1; 1-z)
 *       where:
 *       H1 = Gamma(c) * Gamma(c-a-b) / (Gamma(c-a) * Gamma(c-b))
 *       H2 = Gamma(c) * Gamma(a+b-c) / (Gamma(a) * Gamma(b))
 *
 *       This accelerates convergence when z is near 1, since |1-z| < |z|
 *       means the transformed series converges faster.
 *
 *    c) Linear transformation for |z| >= 1 (DLMF 15.8.2):
 *       2F1(a,b;c;z) = G1 * (-z)^{-a} * 2F1(a, a-c+1; a-b+1; 1/z)
 *                    + G2 * (-z)^{-b} * 2F1(b, b-c+1; b-a+1; 1/z)
 *       where:
 *       G1 = Gamma(c) * Gamma(b-a) / (Gamma(b) * Gamma(c-a))
 *       G2 = Gamma(c) * Gamma(a-b) / (Gamma(a) * Gamma(c-b))
 *
 * 4. SPECIAL HANDLING FOR INTEGER DIFFERENCES:
 *    When a-b is an integer (for DLMF 15.8.2) or c-a-b is an integer
 *    (for DLMF 15.8.4), the standard formulas have poles in the Gamma
 *    functions. Richardson extrapolation with parameter perturbation
 *    is used to compute the limiting values accurately.
 *
 * 5. REAL z > 1 LIMITATION:
 *    For real inputs with z > 1 and non-integer (a-b), the analytic
 *    continuation is generally complex. This implementation computes
 *    the full complex result internally and returns the real part,
 *    which may be incorrect. Use complex inputs for z > 1.
 *
 * 6. APPLICATIONS:
 *    The 2F1 function appears in many contexts:
 *    - Incomplete beta function: I_z(a,b) = z^a / (a*B(a,b)) * 2F1(a, 1-b; a+1; z)
 *    - Legendre functions
 *    - Jacobi polynomials
 *    - Many statistical distributions
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <limits>

#include "gamma.h"
#include "digamma.h"
#include "sin_pi.h"
#include "cos_pi.h"
#include "log_gamma.h"
#include "sign_gamma.h"

namespace torchscience::impl::special_functions {

// Forward declaration for main function (needed throughout the file)
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2_f_1(scalar_t a, scalar_t b, scalar_t c, scalar_t z);

// ============================================================================
// Constants
// ============================================================================

constexpr double kPi_2F1 = 3.14159265358979323846264338327950288;

// ============================================================================
// Series computation
// ============================================================================

/**
 * Gauss hypergeometric series 2F1(a, b; c; z) for |z| < 1.
 *
 * Uses the series expansion:
 *   2F1(a, b; c; z) = sum_{n=0}^{inf} (a)_n * (b)_n / ((c)_n * n!) * z^n
 *
 * where (x)_n = x*(x+1)*...*(x+n-1) is the Pochhammer symbol (rising factorial).
 *
 * Convergence:
 *   - Absolutely convergent for |z| < 1
 *   - Conditionally convergent for |z| = 1 if Re(c - a - b) > 0
 *
 * Special cases:
 *   - When a or b is a non-positive integer -m, the series terminates after m+1 terms
 *     (becomes a polynomial), since (a)_n = 0 for n > m.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_series(scalar_t a, scalar_t b, scalar_t c, scalar_t z) {
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  const int max_terms = 200;
  const real_t eps = std::numeric_limits<real_t>::epsilon() * real_t(100);

  real_t z_mag = abs(z);
  if (z_mag < eps) {
    return scalar_t(1);
  }

  scalar_t sum = scalar_t(1);
  scalar_t term = scalar_t(1);

  for (int n = 0; n < max_terms; ++n) {
    scalar_t a_n = a + scalar_t(n);
    scalar_t b_n = b + scalar_t(n);
    scalar_t c_n = c + scalar_t(n);
    scalar_t n_plus_1 = scalar_t(n + 1);

    real_t c_n_mag = abs(c_n);
    if (c_n_mag < eps) {
      return scalar_t(std::numeric_limits<real_t>::quiet_NaN());
    }

    real_t a_n_mag = abs(a_n);
    real_t b_n_mag = abs(b_n);
    if (a_n_mag < eps || b_n_mag < eps) {
      break;
    }

    term *= (a_n * b_n) / (c_n * n_plus_1) * z;
    sum += term;

    real_t term_mag = abs(term);
    real_t sum_mag = abs(sum);

    if (term_mag < eps * (sum_mag > real_t(1) ? sum_mag : real_t(1))) {
      break;
    }
  }

  return sum;
}

/**
 * Derivative of 2F1 with respect to z.
 *
 * Uses the identity:
 *   d/dz 2F1(a, b; c; z) = (a * b / c) * 2F1(a+1, b+1; c+1; z)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_derivative(scalar_t a, scalar_t b, scalar_t c, scalar_t z) {
  scalar_t coeff = (a * b) / c;
  scalar_t hyp = hypergeometric_2_f_1(a + scalar_t(1), b + scalar_t(1),
                                      c + scalar_t(1), z);
  return coeff * hyp;
}

/**
 * Gauss hypergeometric series with analytical parameter derivatives.
 *
 * Computes 2F1(a, b; c; z) and its derivatives with respect to a, b, c
 * simultaneously in a single pass through the series.
 *
 * The parameter derivatives are computed using:
 *   d/da T_n = T_n * [psi(a+n) - psi(a)] = T_n * sum_{k=0}^{n-1} 1/(a+k)
 *   d/db T_n = T_n * [psi(b+n) - psi(b)] = T_n * sum_{k=0}^{n-1} 1/(b+k)
 *   d/dc T_n = -T_n * [psi(c+n) - psi(c)] = -T_n * sum_{k=0}^{n-1} 1/(c+k)
 *
 * where T_n = (a)_n * (b)_n / ((c)_n * n!) * z^n is the n-th term.
 *
 * Returns: (value, d/da, d/db, d/dc)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
hypergeometric_2f1_series_with_param_derivatives(scalar_t a, scalar_t b, scalar_t c, scalar_t z) {
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  const int max_terms = 200;
  const real_t eps = std::numeric_limits<real_t>::epsilon() * real_t(100);

  real_t z_mag = abs(z);
  if (z_mag < eps) {
    return std::make_tuple(scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(0));
  }

  scalar_t sum = scalar_t(1);
  scalar_t d_sum_a = scalar_t(0);
  scalar_t d_sum_b = scalar_t(0);
  scalar_t d_sum_c = scalar_t(0);

  scalar_t term = scalar_t(1);

  scalar_t psi_a = scalar_t(0);
  scalar_t psi_b = scalar_t(0);
  scalar_t psi_c = scalar_t(0);

  for (int n = 0; n < max_terms; ++n) {
    scalar_t a_n = a + scalar_t(n);
    scalar_t b_n = b + scalar_t(n);
    scalar_t c_n = c + scalar_t(n);
    scalar_t n_plus_1 = scalar_t(n + 1);

    real_t c_n_mag = abs(c_n);
    if (c_n_mag < eps) {
      real_t nan_val = std::numeric_limits<real_t>::quiet_NaN();
      return std::make_tuple(scalar_t(nan_val), scalar_t(nan_val),
                             scalar_t(nan_val), scalar_t(nan_val));
    }

    real_t a_n_mag = abs(a_n);
    real_t b_n_mag = abs(b_n);
    if (a_n_mag < eps || b_n_mag < eps) {
      break;
    }

    psi_a += scalar_t(1) / a_n;
    psi_b += scalar_t(1) / b_n;
    psi_c += scalar_t(1) / c_n;

    term *= (a_n * b_n) / (c_n * n_plus_1) * z;

    sum += term;

    d_sum_a += term * psi_a;
    d_sum_b += term * psi_b;
    d_sum_c -= term * psi_c;

    real_t term_mag = abs(term);
    real_t sum_mag = abs(sum);

    if (term_mag < eps * (sum_mag > real_t(1) ? sum_mag : real_t(1))) {
      break;
    }
  }

  return std::make_tuple(sum, d_sum_a, d_sum_b, d_sum_c);
}

// ============================================================================
// Integer detection
// ============================================================================

/**
 * Check if a scalar value is close to an integer.
 *
 * Returns true if |x - round(x)| < tolerance, along with the nearest integer.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<bool, int>
is_near_integer(scalar_t x) {
  using std::abs;
  using std::round;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  real_t tol;
  if constexpr (std::is_same_v<real_t, double>) {
    tol = real_t(1e-10);
  } else {
    tol = real_t(1e-5);
  }

  real_t x_real;
  if constexpr (c10::is_complex<scalar_t>::value) {
    if (abs(x.imag()) > tol) {
      return std::make_tuple(false, 0);
    }
    x_real = x.real();
  } else {
    x_real = x;
  }

  real_t rounded = round(x_real);
  real_t diff = abs(x_real - rounded);

  if (diff < tol) {
    return std::make_tuple(true, static_cast<int>(rounded));
  }
  return std::make_tuple(false, 0);
}

/**
 * Check if a value is close to a half-integer (n + 0.5).
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE bool
is_near_half_integer(scalar_t x) {
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  real_t tol;
  if constexpr (std::is_same_v<real_t, double>) {
    tol = real_t(1e-10);
  } else {
    tol = real_t(1e-5);
  }

  real_t x_real;
  if constexpr (c10::is_complex<scalar_t>::value) {
    if (abs(x.imag()) > tol) {
      return false;
    }
    x_real = x.real();
  } else {
    x_real = x;
  }

  // Check if x - 0.5 is near an integer
  auto [is_int, n] = is_near_integer(scalar_t(x_real - real_t(0.5)));
  return is_int;
}

// ============================================================================
// Forward declarations for mutual recursion
// ============================================================================

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_linear_transform_integer_diff(
    scalar_t a, scalar_t b, scalar_t c, scalar_t z, int n);

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_one_minus_z_transform_integer_diff(
    scalar_t a, scalar_t b, scalar_t c, scalar_t z, int n);

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
hypergeometric_2f1_linear_transform_integer_diff_with_derivatives(
    scalar_t a, scalar_t b, scalar_t c, scalar_t z, int n);

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
hypergeometric_2f1_one_minus_z_transform_integer_diff_with_derivatives(
    scalar_t a, scalar_t b, scalar_t c, scalar_t z, int n);

// ============================================================================
// 1-z transformation (DLMF 15.8.4)
// ============================================================================

/**
 * Helper function to evaluate DLMF 15.8.4 formula with a given perturbation delta.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_one_minus_z_transform_with_delta(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  typename c10::scalar_value_type<scalar_t>::type delta
) {
  using std::exp;
  using std::log;
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  scalar_t c_perturbed = c + scalar_t(delta);
  scalar_t one_minus_z = scalar_t(1) - z;
  scalar_t cab = c_perturbed - a - b;

  scalar_t power_factor;
  if constexpr (c10::is_complex<scalar_t>::value) {
    power_factor = exp(cab * log(one_minus_z));
  } else {
    if (one_minus_z > real_t(0)) {
      power_factor = exp(cab * log(one_minus_z));
    } else {
      real_t pi_val = real_t(kPi_2F1);
      real_t log_abs = log(abs(one_minus_z));
      power_factor = exp(cab * log_abs) * cos(cab * pi_val);
    }
  }

  scalar_t H1, H2;
  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_H1 = log_gamma_complex(c_perturbed) + log_gamma_complex(cab)
                    - log_gamma_complex(c_perturbed - a) - log_gamma_complex(c_perturbed - b);
    scalar_t log_H2 = log_gamma_complex(c_perturbed) + log_gamma_complex(-cab)
                    - log_gamma_complex(a) - log_gamma_complex(b);
    H1 = exp(log_H1);
    H2 = exp(log_H2);
  } else {
    using std::lgamma;

    scalar_t log_mag_H1 = lgamma(c_perturbed) + lgamma(cab)
                        - lgamma(c_perturbed - a) - lgamma(c_perturbed - b);
    scalar_t log_mag_H2 = lgamma(c_perturbed) + lgamma(-cab)
                        - lgamma(a) - lgamma(b);

    int sign_H1 = sign_gamma(c_perturbed) * sign_gamma(cab)
                * sign_gamma(c_perturbed - a) * sign_gamma(c_perturbed - b);
    int sign_H2 = sign_gamma(c_perturbed) * sign_gamma(-cab)
                * sign_gamma(a) * sign_gamma(b);

    H1 = scalar_t(sign_H1) * exp(log_mag_H1);
    H2 = scalar_t(sign_H2) * exp(log_mag_H2);
  }

  scalar_t series1 = hypergeometric_2f1_series(a, b, a + b - c_perturbed + scalar_t(1), one_minus_z);
  scalar_t series2 = hypergeometric_2f1_series(c_perturbed - a, c_perturbed - b,
                                                cab + scalar_t(1), one_minus_z);

  return H1 * series1 + H2 * power_factor * series2;
}

/**
 * Helper function to evaluate DLMF 15.8.4 with delta and compute derivatives.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
hypergeometric_2f1_one_minus_z_transform_with_delta_and_derivatives(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  typename c10::scalar_value_type<scalar_t>::type delta
) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::cos;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  scalar_t c_perturbed = c + scalar_t(delta);
  scalar_t one_minus_z = scalar_t(1) - z;
  scalar_t cab = c_perturbed - a - b;

  scalar_t log_one_minus_z;
  if constexpr (c10::is_complex<scalar_t>::value) {
    log_one_minus_z = log(one_minus_z);
  } else {
    if (one_minus_z > real_t(0)) {
      log_one_minus_z = log(one_minus_z);
    } else {
      log_one_minus_z = log(abs(one_minus_z));
    }
  }

  scalar_t P;
  if constexpr (c10::is_complex<scalar_t>::value) {
    P = exp(cab * log_one_minus_z);
  } else {
    if (one_minus_z > real_t(0)) {
      P = exp(cab * log_one_minus_z);
    } else {
      real_t pi_val = real_t(kPi_2F1);
      real_t log_abs = log(abs(one_minus_z));
      P = exp(cab * log_abs) * cos(cab * pi_val);
    }
  }

  scalar_t psi_c_p = digamma(c_perturbed);
  scalar_t psi_cab = digamma(cab);
  scalar_t psi_neg_cab = digamma(-cab);
  scalar_t psi_c_p_minus_a = digamma(c_perturbed - a);
  scalar_t psi_c_p_minus_b = digamma(c_perturbed - b);
  scalar_t psi_a = digamma(a);
  scalar_t psi_b = digamma(b);

  scalar_t H1, H2;
  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_H1 = log_gamma_complex(c_perturbed) + log_gamma_complex(cab)
                    - log_gamma_complex(c_perturbed - a) - log_gamma_complex(c_perturbed - b);
    scalar_t log_H2 = log_gamma_complex(c_perturbed) + log_gamma_complex(-cab)
                    - log_gamma_complex(a) - log_gamma_complex(b);
    H1 = exp(log_H1);
    H2 = exp(log_H2);
  } else {
    using std::lgamma;

    scalar_t log_mag_H1 = lgamma(c_perturbed) + lgamma(cab) - lgamma(c_perturbed - a) - lgamma(c_perturbed - b);
    scalar_t log_mag_H2 = lgamma(c_perturbed) + lgamma(-cab) - lgamma(a) - lgamma(b);

    int sign_H1 = sign_gamma(c_perturbed) * sign_gamma(cab)
                * sign_gamma(c_perturbed - a) * sign_gamma(c_perturbed - b);
    int sign_H2 = sign_gamma(c_perturbed) * sign_gamma(-cab)
                * sign_gamma(a) * sign_gamma(b);

    H1 = scalar_t(sign_H1) * exp(log_mag_H1);
    H2 = scalar_t(sign_H2) * exp(log_mag_H2);
  }

  scalar_t dlogH1_da = -psi_cab + psi_c_p_minus_a;
  scalar_t dlogH1_db = -psi_cab + psi_c_p_minus_b;
  scalar_t dlogH1_dc = psi_c_p + psi_cab - psi_c_p_minus_a - psi_c_p_minus_b;

  scalar_t dlogH2_da = psi_neg_cab - psi_a;
  scalar_t dlogH2_db = psi_neg_cab - psi_b;
  scalar_t dlogH2_dc = psi_c_p - psi_neg_cab;

  scalar_t dlogP_da = -log_one_minus_z;
  scalar_t dlogP_db = -log_one_minus_z;
  scalar_t dlogP_dc = log_one_minus_z;

  scalar_t c1 = a + b - c_perturbed + scalar_t(1);
  auto [F1, dF1_da_inner, dF1_db_inner, dF1_dc_inner] =
      hypergeometric_2f1_series_with_param_derivatives(a, b, c1, one_minus_z);

  scalar_t dF1_da = dF1_da_inner + dF1_dc_inner;
  scalar_t dF1_db = dF1_db_inner + dF1_dc_inner;
  scalar_t dF1_dc = -dF1_dc_inner;

  scalar_t a2 = c_perturbed - a;
  scalar_t b2 = c_perturbed - b;
  scalar_t c2 = cab + scalar_t(1);
  auto [F2, dF2_da2, dF2_db2, dF2_dc2] =
      hypergeometric_2f1_series_with_param_derivatives(a2, b2, c2, one_minus_z);

  scalar_t dF2_da = -dF2_da2 - dF2_dc2;
  scalar_t dF2_db = -dF2_db2 - dF2_dc2;
  scalar_t dF2_dc = dF2_da2 + dF2_db2 + dF2_dc2;

  scalar_t term1 = H1 * F1;
  scalar_t term2 = H2 * P * F2;
  scalar_t f = term1 + term2;

  scalar_t df_da = term1 * dlogH1_da + H1 * dF1_da
                 + term2 * dlogH2_da + term2 * dlogP_da + H2 * P * dF2_da;

  scalar_t df_db = term1 * dlogH1_db + H1 * dF1_db
                 + term2 * dlogH2_db + term2 * dlogP_db + H2 * P * dF2_db;

  scalar_t df_dc = term1 * dlogH1_dc + H1 * dF1_dc
                 + term2 * dlogH2_dc + term2 * dlogP_dc + H2 * P * dF2_dc;

  return std::make_tuple(f, df_da, df_db, df_dc);
}

/**
 * Explicit limiting form of 1-z transformation when c-a-b is a non-negative integer m.
 * Uses DLMF 15.8.10 formula instead of Richardson extrapolation for higher precision.
 *
 * For c = a + b + m (equivalently c - a - b = m >= 0), |1-z| < 1:
 *
 * ₂F₁(a, b; a+b+m; z) = [Γ(a+b+m)/(Γ(a+m)Γ(b+m))] × {
 *   Σ_{k=0}^{m-1} [(a)_k (b)_k / k!] × (m-1-k)! × (1-z)^k
 *   + [(-1)^m (1-z)^m / (m-1)!] × Σ_{s=0}^∞ [(a+m)_s (b+m)_s / (s!(s+m)!)] × (1-z)^s
 *     × [ψ(s+1) + ψ(s+m+1) - ψ(a+m+s) - ψ(b+m+s) - ln(1-z)]
 * }
 *
 * For m = 0 (c = a + b), the polynomial sum is empty and the formula simplifies.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_one_minus_z_transform_integer_diff_explicit(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  int m
) {
  using std::exp;
  using std::log;
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  const int max_terms = 200;
  const real_t eps = std::numeric_limits<real_t>::epsilon() * real_t(100);

  scalar_t one_minus_z = scalar_t(1) - z;

  // Special case: if a or b is zero (or negative integer), the series terminates
  // In this case, the standard limiting formula isn't needed - just use the series directly
  // Note: ₂F₁(a, b; c; z) with b=0 equals 1, with b=-n equals a polynomial
  auto [a_is_int, a_int] = is_near_integer(a);
  auto [b_is_int, b_int] = is_near_integer(b);
  if ((a_is_int && a_int <= 0) || (b_is_int && b_int <= 0)) {
    // For terminating series, use the regular series formula
    // The explicit formula here computes ₂F₁(a, b; c; z) where c = a + b + m
    // and the series argument is z (not 1-z)
    return hypergeometric_2f1_series(a, b, c, z);
  }

  // Compute ln(1-z)
  scalar_t log_one_minus_z;
  if constexpr (c10::is_complex<scalar_t>::value) {
    log_one_minus_z = log(one_minus_z);
  } else {
    real_t omz_val = static_cast<real_t>(one_minus_z);
    if (omz_val > real_t(0)) {
      log_one_minus_z = log(omz_val);
    } else {
      // 1-z < 0: log(1-z) = log|1-z| + i*pi
      // For real inputs, we use the real part
      log_one_minus_z = log(abs(omz_val));
    }
  }

  // Compute Gamma ratio: Γ(a+b+m) / (Γ(a+m)Γ(b+m))
  // Note: c = a + b + m, so Γ(c) / (Γ(a+m)Γ(b+m)) = Γ(a+b+m) / (Γ(a+m)Γ(b+m))
  scalar_t a_m = a + scalar_t(m);
  scalar_t b_m = b + scalar_t(m);

  scalar_t gamma_ratio_coeff;
  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_ratio = log_gamma_complex(c) - log_gamma_complex(a_m) - log_gamma_complex(b_m);
    gamma_ratio_coeff = exp(log_ratio);
  } else {
    using std::lgamma;
    scalar_t log_mag = lgamma(c) - lgamma(a_m) - lgamma(b_m);
    int sign = sign_gamma(c) * sign_gamma(a_m) * sign_gamma(b_m);
    gamma_ratio_coeff = scalar_t(sign) * exp(log_mag);
  }

  scalar_t result = scalar_t(0);

  // =========================================================================
  // Part 1: Finite polynomial sum (only for m >= 1)
  // Σ_{k=0}^{m-1} [(a)_k (b)_k / k!] × (m-1-k)! × (1-z)^k
  // =========================================================================
  if (m >= 1) {
    scalar_t poly_sum = scalar_t(0);
    scalar_t poch_a = scalar_t(1);       // (a)_k
    scalar_t poch_b = scalar_t(1);       // (b)_k
    scalar_t omz_pow = scalar_t(1);      // (1-z)^k
    scalar_t factorial_k = scalar_t(1);  // k!

    for (int k = 0; k < m; ++k) {
      // (m-1-k)!
      real_t factorial_mmk = real_t(1);
      for (int j = 2; j <= m - 1 - k; ++j) {
        factorial_mmk *= real_t(j);
      }

      scalar_t term = (poch_a * poch_b / factorial_k) * scalar_t(factorial_mmk) * omz_pow;
      poly_sum += term;

      // Update for next iteration
      if (k < m - 1) {
        poch_a *= (a + scalar_t(k));
        poch_b *= (b + scalar_t(k));
        omz_pow *= one_minus_z;
        factorial_k *= scalar_t(k + 1);
      }
    }

    result += poly_sum;
  }

  // =========================================================================
  // Part 2: Infinite series with logarithmic terms
  // [(-1)^m (1-z)^m / (m-1)!] × Σ_{s=0}^∞ [(a+m)_s (b+m)_s / (s!(s+m)!)] × (1-z)^s
  //   × [ψ(s+1) + ψ(s+m+1) - ψ(a+m+s) - ψ(b+m+s) - ln(1-z)]
  // =========================================================================
  {
    // Compute (-1)^m / (m-1)! for m >= 1, or handle m = 0 specially
    scalar_t series_coeff;
    if (m == 0) {
      // For m = 0, coefficient is 1 and digamma combination is
      // [2ψ(s+1) - ψ(a+s) - ψ(b+s) - ln(1-z)]
      series_coeff = scalar_t(1);
    } else {
      // (-1)^m / (m-1)!
      real_t factorial_mm1 = real_t(1);
      for (int j = 2; j <= m - 1; ++j) {
        factorial_mm1 *= real_t(j);
      }
      series_coeff = scalar_t((m % 2 == 0) ? 1 : -1) / scalar_t(factorial_mm1);
    }

    // Compute (1-z)^m
    scalar_t omz_m = scalar_t(1);
    for (int j = 0; j < m; ++j) {
      omz_m *= one_minus_z;
    }

    scalar_t log_series = scalar_t(0);
    scalar_t poch_am = scalar_t(1);      // (a+m)_s
    scalar_t poch_bm = scalar_t(1);      // (b+m)_s
    scalar_t omz_pow = scalar_t(1);      // (1-z)^s

    // Accumulate digamma sums for incremental computation
    scalar_t psi_sum_am = scalar_t(0);   // Σ 1/(a+m+j) from j=0 to s-1
    scalar_t psi_sum_bm = scalar_t(0);   // Σ 1/(b+m+j) from j=0 to s-1

    // Base digamma values
    scalar_t psi_am = digamma(a_m);
    scalar_t psi_bm = digamma(b_m);

    for (int s = 0; s < max_terms; ++s) {
      // Compute factorials: s! and (s+m)!
      real_t factorial_s = real_t(1);
      for (int j = 2; j <= s; ++j) {
        factorial_s *= real_t(j);
      }
      real_t factorial_sm = factorial_s;
      for (int j = s + 1; j <= s + m; ++j) {
        factorial_sm *= real_t(j);
      }

      // Digamma combination
      scalar_t psi_s1 = digamma(scalar_t(s + 1));        // ψ(s+1)
      scalar_t psi_sm1 = digamma(scalar_t(s + m + 1));   // ψ(s+m+1)
      scalar_t psi_ams = psi_am + psi_sum_am;            // ψ(a+m+s)
      scalar_t psi_bms = psi_bm + psi_sum_bm;            // ψ(b+m+s)

      scalar_t H;
      if (m == 0) {
        // For m = 0: [2ψ(s+1) - ψ(a+s) - ψ(b+s) - ln(1-z)]
        H = scalar_t(2) * psi_s1 - psi_ams - psi_bms - log_one_minus_z;
      } else {
        // For m >= 1: [ψ(s+1) + ψ(s+m+1) - ψ(a+m+s) - ψ(b+m+s) - ln(1-z)]
        H = psi_s1 + psi_sm1 - psi_ams - psi_bms - log_one_minus_z;
      }

      scalar_t term = (poch_am * poch_bm) / scalar_t(factorial_s * factorial_sm) * omz_pow * H;
      log_series += term;

      // Check convergence
      real_t term_mag = abs(term);
      real_t sum_mag = abs(log_series);
      if (term_mag < eps * (sum_mag > real_t(1) ? sum_mag : real_t(1))) {
        break;
      }

      // Update Pochhammer symbols and power for next iteration
      psi_sum_am += scalar_t(1) / (a_m + scalar_t(s));
      psi_sum_bm += scalar_t(1) / (b_m + scalar_t(s));
      poch_am *= (a_m + scalar_t(s));
      poch_bm *= (b_m + scalar_t(s));
      omz_pow *= one_minus_z;
    }

    result += series_coeff * omz_m * log_series;
  }

  return gamma_ratio_coeff * result;
}

/**
 * Limiting form of 1-z transformation when c-a-b is an integer m.
 * Dispatches to explicit formula or uses Euler transformation for negative m.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_one_minus_z_transform_integer_diff(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  int m
) {
  using std::abs;
  using std::exp;
  using std::log;
  using std::cos;
  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  // For m < 0 (c - a - b < 0), use Euler's transformation (DLMF 15.8.1):
  // ₂F₁(a, b; c; z) = (1-z)^{c-a-b} × ₂F₁(c-a, c-b; c; z)
  // The transformed function has c - (c-a) - (c-b) = a + b - c = -m > 0
  if (m < 0) {
    scalar_t one_minus_z = scalar_t(1) - z;
    scalar_t cab = c - a - b;  // = m < 0

    // Compute (1-z)^{c-a-b}
    scalar_t power_factor;
    if constexpr (c10::is_complex<scalar_t>::value) {
      power_factor = exp(cab * log(one_minus_z));
    } else {
      real_t omz_val = static_cast<real_t>(one_minus_z);
      if (omz_val > real_t(0)) {
        power_factor = exp(cab * log(omz_val));
      } else {
        real_t pi_val = real_t(kPi_2F1);
        power_factor = exp(cab * log(abs(omz_val))) * cos(cab * pi_val);
      }
    }

    // Apply transformed parameters: a_new = c - a, b_new = c - b
    scalar_t a_new = c - a;
    scalar_t b_new = c - b;
    int m_new = -m;  // Now c - a_new - b_new = -m > 0

    // Call recursively with positive m
    return power_factor * hypergeometric_2f1_one_minus_z_transform_integer_diff(
        a_new, b_new, c, z, m_new);
  }

  // For m >= 0, use Richardson extrapolation
  // This is more robust than the explicit limiting formula for general cases
  real_t delta1, delta2;
  if constexpr (std::is_same_v<real_t, double>) {
    delta1 = real_t(1e-7);
    delta2 = real_t(5e-8);
  } else {
    delta1 = real_t(1e-4);
    delta2 = real_t(5e-5);
  }

  scalar_t f1 = hypergeometric_2f1_one_minus_z_transform_with_delta(a, b, c, z, delta1);
  scalar_t f2 = hypergeometric_2f1_one_minus_z_transform_with_delta(a, b, c, z, delta2);

  return (f1 * scalar_t(delta2) - f2 * scalar_t(delta1)) / scalar_t(delta2 - delta1);
}

/**
 * Explicit derivative of 1-z transformation when c-a-b is a non-negative integer m.
 *
 * Computes the value and derivatives with respect to a, b, c analytically
 * using the DLMF 15.8.10 formula structure.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
hypergeometric_2f1_one_minus_z_transform_integer_diff_explicit_with_derivatives(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  int m
) {
  using std::exp;
  using std::log;
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  const int max_terms = 200;
  const real_t eps = std::numeric_limits<real_t>::epsilon() * real_t(100);

  scalar_t one_minus_z = scalar_t(1) - z;

  // Compute ln(1-z)
  scalar_t log_one_minus_z;
  if constexpr (c10::is_complex<scalar_t>::value) {
    log_one_minus_z = log(one_minus_z);
  } else {
    real_t omz_val = static_cast<real_t>(one_minus_z);
    if (omz_val > real_t(0)) {
      log_one_minus_z = log(omz_val);
    } else {
      log_one_minus_z = log(abs(omz_val));
    }
  }

  // Parameters for the formula
  scalar_t a_m = a + scalar_t(m);
  scalar_t b_m = b + scalar_t(m);

  // Gamma ratio: G = Γ(c) / (Γ(a+m)Γ(b+m))
  // dlogG/da = -ψ(a+m), dlogG/db = -ψ(b+m), dlogG/dc = ψ(c)
  scalar_t psi_c = digamma(c);
  scalar_t psi_am = digamma(a_m);
  scalar_t psi_bm = digamma(b_m);

  scalar_t gamma_ratio;
  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_ratio = log_gamma_complex(c) - log_gamma_complex(a_m) - log_gamma_complex(b_m);
    gamma_ratio = exp(log_ratio);
  } else {
    using std::lgamma;
    scalar_t log_mag = lgamma(c) - lgamma(a_m) - lgamma(b_m);
    int sign = sign_gamma(c) * sign_gamma(a_m) * sign_gamma(b_m);
    gamma_ratio = scalar_t(sign) * exp(log_mag);
  }

  scalar_t dlogG_da = -psi_am;
  scalar_t dlogG_db = -psi_bm;
  scalar_t dlogG_dc = psi_c;

  // Initialize sums for value and derivatives
  scalar_t S = scalar_t(0);      // The bracketed expression
  scalar_t dS_da = scalar_t(0);
  scalar_t dS_db = scalar_t(0);
  scalar_t dS_dc = scalar_t(0);  // Will be 0 since polynomial/series don't depend on c directly

  // =========================================================================
  // Part 1: Finite polynomial sum (only for m >= 1)
  // P = Σ_{k=0}^{m-1} [(a)_k (b)_k / k!] × (m-1-k)! × (1-z)^k
  // dP/da = Σ_{k=0}^{m-1} [d(a)_k/da × (b)_k / k!] × (m-1-k)! × (1-z)^k
  //       where d(a)_k/da = (a)_k × Σ_{j=0}^{k-1} 1/(a+j)
  // =========================================================================
  if (m >= 1) {
    scalar_t poch_a = scalar_t(1);
    scalar_t poch_b = scalar_t(1);
    scalar_t omz_pow = scalar_t(1);
    scalar_t factorial_k = scalar_t(1);
    scalar_t psi_sum_a = scalar_t(0);  // Σ 1/(a+j)
    scalar_t psi_sum_b = scalar_t(0);  // Σ 1/(b+j)

    for (int k = 0; k < m; ++k) {
      real_t factorial_mmk = real_t(1);
      for (int j = 2; j <= m - 1 - k; ++j) {
        factorial_mmk *= real_t(j);
      }

      scalar_t base_term = (poch_a * poch_b / factorial_k) * scalar_t(factorial_mmk) * omz_pow;
      S += base_term;

      // Derivatives: d[(a)_k]/da = (a)_k × psi_sum_a, similarly for b
      if (k > 0) {
        dS_da += base_term * psi_sum_a;
        dS_db += base_term * psi_sum_b;
      }

      // Update for next iteration
      if (k < m - 1) {
        psi_sum_a += scalar_t(1) / (a + scalar_t(k));
        psi_sum_b += scalar_t(1) / (b + scalar_t(k));
        poch_a *= (a + scalar_t(k));
        poch_b *= (b + scalar_t(k));
        omz_pow *= one_minus_z;
        factorial_k *= scalar_t(k + 1);
      }
    }
  }

  // =========================================================================
  // Part 2: Infinite series with logarithmic terms
  // L = coeff × (1-z)^m × Σ_{s=0}^∞ [(a+m)_s (b+m)_s / (s!(s+m)!)] × (1-z)^s × H_s
  // where H_s = ψ(s+1) + ψ(s+m+1) - ψ(a+m+s) - ψ(b+m+s) - ln(1-z)
  // =========================================================================
  {
    scalar_t series_coeff;
    if (m == 0) {
      series_coeff = scalar_t(1);
    } else {
      real_t factorial_mm1 = real_t(1);
      for (int j = 2; j <= m - 1; ++j) {
        factorial_mm1 *= real_t(j);
      }
      series_coeff = scalar_t((m % 2 == 0) ? 1 : -1) / scalar_t(factorial_mm1);
    }

    scalar_t omz_m = scalar_t(1);
    for (int j = 0; j < m; ++j) {
      omz_m *= one_minus_z;
    }

    scalar_t L = scalar_t(0);
    scalar_t dL_da = scalar_t(0);
    scalar_t dL_db = scalar_t(0);

    scalar_t poch_am = scalar_t(1);
    scalar_t poch_bm = scalar_t(1);
    scalar_t omz_pow = scalar_t(1);
    scalar_t psi_sum_am = scalar_t(0);
    scalar_t psi_sum_bm = scalar_t(0);

    // Base digamma values
    scalar_t psi_am_base = digamma(a_m);
    scalar_t psi_bm_base = digamma(b_m);

    for (int s = 0; s < max_terms; ++s) {
      real_t factorial_s = real_t(1);
      for (int j = 2; j <= s; ++j) {
        factorial_s *= real_t(j);
      }
      real_t factorial_sm = factorial_s;
      for (int j = s + 1; j <= s + m; ++j) {
        factorial_sm *= real_t(j);
      }

      scalar_t psi_s1 = digamma(scalar_t(s + 1));
      scalar_t psi_sm1 = digamma(scalar_t(s + m + 1));
      scalar_t psi_ams = psi_am_base + psi_sum_am;
      scalar_t psi_bms = psi_bm_base + psi_sum_bm;

      scalar_t H;
      if (m == 0) {
        H = scalar_t(2) * psi_s1 - psi_ams - psi_bms - log_one_minus_z;
      } else {
        H = psi_s1 + psi_sm1 - psi_ams - psi_bms - log_one_minus_z;
      }

      scalar_t coeff_term = (poch_am * poch_bm) / scalar_t(factorial_s * factorial_sm) * omz_pow;
      scalar_t term = coeff_term * H;
      L += term;

      // Derivatives of the series term
      // d/da of coeff_term = coeff_term × psi_sum_am (from d(a+m)_s/da)
      // d/da of H = -d(psi(a+m+s))/da = -trigamma(a+m+s) (but we approximate via psi_sum)
      // Simplified: d/da [coeff_term × H] ≈ coeff_term × (H × psi_sum_am - 1/(a+m+s) sum)

      // For simplicity, we use the chain rule:
      // dL/da = Σ [d(poch_am)/da × rest + poch_am × d(H)/da]
      // d(poch_am)/da = poch_am × psi_sum_am
      // dH/da = -dψ(a+m+s)/da ≈ contribution handled via recurrence

      // Approximate analytical derivative using partial derivatives
      scalar_t trigamma_ams = scalar_t(0);  // Second derivative of lgamma
      for (int j = 0; j <= s; ++j) {
        trigamma_ams += scalar_t(1) / ((a_m + scalar_t(j)) * (a_m + scalar_t(j)));
      }
      scalar_t trigamma_bms = scalar_t(0);
      for (int j = 0; j <= s; ++j) {
        trigamma_bms += scalar_t(1) / ((b_m + scalar_t(j)) * (b_m + scalar_t(j)));
      }

      scalar_t dH_da = -trigamma_ams;  // Approximate: -ψ'(a+m+s) ≈ -Σ 1/(a+m+j)²
      scalar_t dH_db = -trigamma_bms;

      dL_da += coeff_term * (H * psi_sum_am + dH_da);
      dL_db += coeff_term * (H * psi_sum_bm + dH_db);

      // Convergence check
      real_t term_mag = abs(term);
      real_t sum_mag = abs(L);
      if (term_mag < eps * (sum_mag > real_t(1) ? sum_mag : real_t(1))) {
        break;
      }

      // Update for next iteration
      psi_sum_am += scalar_t(1) / (a_m + scalar_t(s));
      psi_sum_bm += scalar_t(1) / (b_m + scalar_t(s));
      poch_am *= (a_m + scalar_t(s));
      poch_bm *= (b_m + scalar_t(s));
      omz_pow *= one_minus_z;
    }

    S += series_coeff * omz_m * L;
    dS_da += series_coeff * omz_m * dL_da;
    dS_db += series_coeff * omz_m * dL_db;
  }

  // Final result: f = G × S
  // df/da = G × dS/da + S × G × dlogG/da
  // df/db = G × dS/db + S × G × dlogG/db
  // df/dc = S × G × dlogG/dc
  scalar_t f = gamma_ratio * S;
  scalar_t df_da = gamma_ratio * (dS_da + S * dlogG_da);
  scalar_t df_db = gamma_ratio * (dS_db + S * dlogG_db);
  scalar_t df_dc = gamma_ratio * (S * dlogG_dc);

  return std::make_tuple(f, df_da, df_db, df_dc);
}

/**
 * Limiting form of 1-z transformation with derivatives.
 * Dispatches to explicit formula or uses Euler transformation for negative m.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
hypergeometric_2f1_one_minus_z_transform_integer_diff_with_derivatives(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  int m
) {
  using std::abs;
  using std::exp;
  using std::log;
  using std::cos;
  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  // For m < 0 (c - a - b < 0), use Euler's transformation (DLMF 15.8.1):
  // ₂F₁(a, b; c; z) = (1-z)^{c-a-b} × ₂F₁(c-a, c-b; c; z)
  if (m < 0) {
    scalar_t one_minus_z = scalar_t(1) - z;
    scalar_t cab = c - a - b;  // = m < 0

    scalar_t log_omz;
    scalar_t power_factor;
    if constexpr (c10::is_complex<scalar_t>::value) {
      log_omz = log(one_minus_z);
      power_factor = exp(cab * log_omz);
    } else {
      real_t omz_val = static_cast<real_t>(one_minus_z);
      if (omz_val > real_t(0)) {
        log_omz = log(omz_val);
        power_factor = exp(cab * log_omz);
      } else {
        real_t pi_val = real_t(kPi_2F1);
        log_omz = log(abs(omz_val));
        power_factor = exp(cab * log_omz) * cos(cab * pi_val);
      }
    }

    // Apply transformed parameters
    scalar_t a_new = c - a;
    scalar_t b_new = c - b;
    int m_new = -m;

    // Get derivatives for transformed function
    auto [f_inner, df_da_new, df_db_new, df_dc_new] =
        hypergeometric_2f1_one_minus_z_transform_integer_diff_with_derivatives(
            a_new, b_new, c, z, m_new);

    // f = P × f_inner where P = (1-z)^{c-a-b}
    // dP/da = -P × ln(1-z), dP/db = -P × ln(1-z), dP/dc = P × ln(1-z)
    // Chain rule for f_inner(c-a, c-b, c, z):
    // df_inner/da|original = -df_inner/da_new
    // df_inner/db|original = -df_inner/db_new
    // df_inner/dc|original = df_da_new + df_db_new + df_dc_new
    scalar_t f = power_factor * f_inner;
    scalar_t df_da = power_factor * (-log_omz * f_inner - df_da_new);
    scalar_t df_db = power_factor * (-log_omz * f_inner - df_db_new);
    scalar_t df_dc = power_factor * (log_omz * f_inner + df_da_new + df_db_new + df_dc_new);

    return std::make_tuple(f, df_da, df_db, df_dc);
  }

  // For m >= 0, use Richardson extrapolation
  real_t delta1, delta2;
  if constexpr (std::is_same_v<real_t, double>) {
    delta1 = real_t(1e-7);
    delta2 = real_t(5e-8);
  } else {
    delta1 = real_t(1e-4);
    delta2 = real_t(5e-5);
  }

  auto [f1, da1, db1, dc1] = hypergeometric_2f1_one_minus_z_transform_with_delta_and_derivatives(a, b, c, z, delta1);
  auto [f2, da2, db2, dc2] = hypergeometric_2f1_one_minus_z_transform_with_delta_and_derivatives(a, b, c, z, delta2);

  scalar_t denom = scalar_t(delta2 - delta1);
  scalar_t w1 = scalar_t(delta2) / denom;
  scalar_t w2 = scalar_t(delta1) / denom;

  scalar_t f = f1 * w1 - f2 * w2;
  scalar_t df_da = da1 * w1 - da2 * w2;
  scalar_t df_db = db1 * w1 - db2 * w2;
  scalar_t df_dc = dc1 * w1 - dc2 * w2;

  return std::make_tuple(f, df_da, df_db, df_dc);
}

/**
 * Transformation for 2F1 when z is near 1 (DLMF 15.8.4).
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_one_minus_z_transform(scalar_t a, scalar_t b, scalar_t c, scalar_t z) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::cos;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  scalar_t cab = c - a - b;

  auto [is_int, n] = is_near_integer(cab);
  if (is_int) {
    return hypergeometric_2f1_one_minus_z_transform_integer_diff(a, b, c, z, n);
  }

  scalar_t one_minus_z = scalar_t(1) - z;

  scalar_t power_factor;
  if constexpr (c10::is_complex<scalar_t>::value) {
    power_factor = exp(cab * log(one_minus_z));
  } else {
    if (one_minus_z > real_t(0)) {
      power_factor = exp(cab * log(one_minus_z));
    } else {
      real_t pi_val = real_t(kPi_2F1);
      real_t log_abs = log(abs(one_minus_z));
      power_factor = exp(cab * log_abs) * cos(cab * pi_val);
    }
  }

  scalar_t H1, H2;
  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_H1 = log_gamma_complex(c) + log_gamma_complex(cab)
                    - log_gamma_complex(c - a) - log_gamma_complex(c - b);
    scalar_t log_H2 = log_gamma_complex(c) + log_gamma_complex(-cab)
                    - log_gamma_complex(a) - log_gamma_complex(b);
    H1 = exp(log_H1);
    H2 = exp(log_H2);
  } else {
    using std::lgamma;

    scalar_t log_mag_H1 = lgamma(c) + lgamma(cab) - lgamma(c - a) - lgamma(c - b);
    scalar_t log_mag_H2 = lgamma(c) + lgamma(-cab) - lgamma(a) - lgamma(b);

    int sign_H1 = sign_gamma(c) * sign_gamma(cab) * sign_gamma(c - a) * sign_gamma(c - b);
    int sign_H2 = sign_gamma(c) * sign_gamma(-cab) * sign_gamma(a) * sign_gamma(b);

    H1 = scalar_t(sign_H1) * exp(log_mag_H1);
    H2 = scalar_t(sign_H2) * exp(log_mag_H2);
  }

  scalar_t series1 = hypergeometric_2f1_series(a, b, a + b - c + scalar_t(1), one_minus_z);
  scalar_t series2 = hypergeometric_2f1_series(c - a, c - b, cab + scalar_t(1), one_minus_z);

  return H1 * series1 + H2 * power_factor * series2;
}

/**
 * 1-z transformation with analytical parameter derivatives.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
hypergeometric_2f1_one_minus_z_transform_with_param_derivatives(
    scalar_t a, scalar_t b, scalar_t c, scalar_t z
) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::cos;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  scalar_t cab = c - a - b;
  scalar_t one_minus_z = scalar_t(1) - z;

  auto [is_int, n] = is_near_integer(cab);
  if (is_int) {
    return hypergeometric_2f1_one_minus_z_transform_integer_diff_with_derivatives(a, b, c, z, n);
  }

  scalar_t log_one_minus_z;
  if constexpr (c10::is_complex<scalar_t>::value) {
    log_one_minus_z = log(one_minus_z);
  } else {
    if (one_minus_z > real_t(0)) {
      log_one_minus_z = log(one_minus_z);
    } else {
      log_one_minus_z = log(abs(one_minus_z));
    }
  }

  scalar_t P;
  if constexpr (c10::is_complex<scalar_t>::value) {
    P = exp(cab * log_one_minus_z);
  } else {
    if (one_minus_z > real_t(0)) {
      P = exp(cab * log_one_minus_z);
    } else {
      real_t pi_val = real_t(kPi_2F1);
      real_t log_abs = log(abs(one_minus_z));
      P = exp(cab * log_abs) * cos(cab * pi_val);
    }
  }

  scalar_t H1, H2;
  scalar_t psi_c, psi_cab, psi_neg_cab, psi_c_minus_a, psi_c_minus_b, psi_a, psi_b;

  psi_c = digamma(c);
  psi_cab = digamma(cab);
  psi_neg_cab = digamma(-cab);
  psi_c_minus_a = digamma(c - a);
  psi_c_minus_b = digamma(c - b);
  psi_a = digamma(a);
  psi_b = digamma(b);

  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_H1 = log_gamma_complex(c) + log_gamma_complex(cab)
                    - log_gamma_complex(c - a) - log_gamma_complex(c - b);
    scalar_t log_H2 = log_gamma_complex(c) + log_gamma_complex(-cab)
                    - log_gamma_complex(a) - log_gamma_complex(b);
    H1 = exp(log_H1);
    H2 = exp(log_H2);
  } else {
    using std::lgamma;

    scalar_t log_mag_H1 = lgamma(c) + lgamma(cab) - lgamma(c - a) - lgamma(c - b);
    scalar_t log_mag_H2 = lgamma(c) + lgamma(-cab) - lgamma(a) - lgamma(b);

    int sign_H1 = sign_gamma(c) * sign_gamma(cab) * sign_gamma(c - a) * sign_gamma(c - b);
    int sign_H2 = sign_gamma(c) * sign_gamma(-cab) * sign_gamma(a) * sign_gamma(b);

    H1 = scalar_t(sign_H1) * exp(log_mag_H1);
    H2 = scalar_t(sign_H2) * exp(log_mag_H2);
  }

  scalar_t dlogH1_da = -psi_cab + psi_c_minus_a;
  scalar_t dlogH1_db = -psi_cab + psi_c_minus_b;
  scalar_t dlogH1_dc = psi_c + psi_cab - psi_c_minus_a - psi_c_minus_b;

  scalar_t dlogH2_da = psi_neg_cab - psi_a;
  scalar_t dlogH2_db = psi_neg_cab - psi_b;
  scalar_t dlogH2_dc = psi_c - psi_neg_cab;

  scalar_t dlogP_da = -log_one_minus_z;
  scalar_t dlogP_db = -log_one_minus_z;
  scalar_t dlogP_dc = log_one_minus_z;

  scalar_t c1 = a + b - c + scalar_t(1);
  auto [F1, dF1_da_inner, dF1_db_inner, dF1_dc_inner] =
      hypergeometric_2f1_series_with_param_derivatives(a, b, c1, one_minus_z);

  scalar_t dF1_da = dF1_da_inner + dF1_dc_inner;
  scalar_t dF1_db = dF1_db_inner + dF1_dc_inner;
  scalar_t dF1_dc = -dF1_dc_inner;

  scalar_t a2 = c - a;
  scalar_t b2 = c - b;
  scalar_t c2 = cab + scalar_t(1);
  auto [F2, dF2_da2, dF2_db2, dF2_dc2] =
      hypergeometric_2f1_series_with_param_derivatives(a2, b2, c2, one_minus_z);

  scalar_t dF2_da = -dF2_da2 - dF2_dc2;
  scalar_t dF2_db = -dF2_db2 - dF2_dc2;
  scalar_t dF2_dc = dF2_da2 + dF2_db2 + dF2_dc2;

  scalar_t term1 = H1 * F1;
  scalar_t term2 = H2 * P * F2;
  scalar_t f = term1 + term2;

  scalar_t df_da = term1 * dlogH1_da + H1 * dF1_da
                 + term2 * dlogH2_da + term2 * dlogP_da + H2 * P * dF2_da;

  scalar_t df_db = term1 * dlogH1_db + H1 * dF1_db
                 + term2 * dlogH2_db + term2 * dlogP_db + H2 * P * dF2_db;

  scalar_t df_dc = term1 * dlogH1_dc + H1 * dF1_dc
                 + term2 * dlogH2_dc + term2 * dlogP_dc + H2 * P * dF2_dc;

  return std::make_tuple(f, df_da, df_db, df_dc);
}

// ============================================================================
// Linear transformation (DLMF 15.8.2)
// ============================================================================

/**
 * Compute (-z)^{-a} for the linear transformation.
 *
 * For complex z or real z < 0: straightforward computation.
 * For real z > 0: (-z)^{-a} = |z|^{-a} * exp(-i*a*pi), which is complex.
 *   We return the real part |z|^{-a} * cos(a*pi).
 *   If a is a half-integer (cos(a*pi) = 0), the result is purely imaginary
 *   and we return NaN to indicate the result cannot be represented as real.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
compute_neg_z_power(scalar_t z, scalar_t exponent) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::cos;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  if constexpr (c10::is_complex<scalar_t>::value) {
    return exp(-exponent * log(-z));
  } else {
    real_t z_abs = abs(z);

    if (z < real_t(0)) {
      // -z > 0, so log(-z) is real
      return exp(-exponent * log(z_abs));
    } else {
      // z > 0, -z < 0, so log(-z) = log|z| + i*pi
      // (-z)^{-a} = |z|^{-a} * exp(-i*a*pi) = |z|^{-a} * (cos(a*pi) - i*sin(a*pi))
      // Return real part: |z|^{-a} * cos(a*pi)
      real_t pi_val = real_t(kPi_2F1);
      real_t pow_factor = exp(-exponent * log(z_abs));
      real_t cos_factor = cos(exponent * pi_val);

      // Check for half-integer exponent where cos(a*pi) = 0
      // In this case, the result is purely imaginary and can't be represented as real
      if (is_near_half_integer(exponent)) {
        return scalar_t(std::numeric_limits<real_t>::quiet_NaN());
      }

      return pow_factor * cos_factor;
    }
  }
}

/**
 * Compute d/da [(-z)^{-a}] / (-z)^{-a} = -log(-z)
 *
 * For real z > 0, this requires special handling since log(-z) is complex.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
compute_neg_z_power_log_derivative(scalar_t z, scalar_t exponent) {
  using std::log;
  using std::abs;
  using std::sin;
  using std::cos;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  if constexpr (c10::is_complex<scalar_t>::value) {
    return -log(-z);
  } else {
    real_t z_abs = abs(z);

    if (z < real_t(0)) {
      return -log(z_abs);
    } else {
      // For z > 0: P(a) = |z|^{-a} * cos(a*pi)
      // dP/da = |z|^{-a} * (-log|z| * cos(a*pi) - pi * sin(a*pi))
      // d(log P)/da = dP/da / P = -log|z| - pi * tan(a*pi)
      real_t pi_val = real_t(kPi_2F1);
      real_t log_z = log(z_abs);
      real_t tan_factor = sin(exponent * pi_val) / cos(exponent * pi_val);
      return -log_z - scalar_t(pi_val) * tan_factor;
    }
  }
}

/**
 * Helper function to evaluate DLMF 15.8.2 formula with a given perturbation delta.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_linear_transform_with_delta(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  typename c10::scalar_value_type<scalar_t>::type delta
) {
  using std::exp;
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  scalar_t a_perturbed = a + scalar_t(delta);
  scalar_t z_inv = scalar_t(1) / z;

  scalar_t factor_a = compute_neg_z_power(z, a_perturbed);
  scalar_t factor_b = compute_neg_z_power(z, b);

  scalar_t G1, G2;
  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_G1 = log_gamma_complex(c) + log_gamma_complex(b - a_perturbed)
                    - log_gamma_complex(b) - log_gamma_complex(c - a_perturbed);
    scalar_t log_G2 = log_gamma_complex(c) + log_gamma_complex(a_perturbed - b)
                    - log_gamma_complex(a_perturbed) - log_gamma_complex(c - b);
    G1 = exp(log_G1);
    G2 = exp(log_G2);
  } else {
    using std::lgamma;

    scalar_t log_mag_G1 = lgamma(c) + lgamma(b - a_perturbed) - lgamma(b) - lgamma(c - a_perturbed);
    scalar_t log_mag_G2 = lgamma(c) + lgamma(a_perturbed - b) - lgamma(a_perturbed) - lgamma(c - b);

    int sign_G1 = sign_gamma(c) * sign_gamma(b - a_perturbed)
                * sign_gamma(b) * sign_gamma(c - a_perturbed);
    int sign_G2 = sign_gamma(c) * sign_gamma(a_perturbed - b)
                * sign_gamma(a_perturbed) * sign_gamma(c - b);

    G1 = scalar_t(sign_G1) * exp(log_mag_G1);
    G2 = scalar_t(sign_G2) * exp(log_mag_G2);
  }

  scalar_t series1 = hypergeometric_2f1_series(
    a_perturbed, a_perturbed - c + scalar_t(1), a_perturbed - b + scalar_t(1), z_inv);
  scalar_t series2 = hypergeometric_2f1_series(
    b, b - c + scalar_t(1), b - a_perturbed + scalar_t(1), z_inv);

  return G1 * factor_a * series1 + G2 * factor_b * series2;
}

/**
 * Helper function to evaluate DLMF 15.8.2 with delta and compute derivatives.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
hypergeometric_2f1_linear_transform_with_delta_and_derivatives(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  typename c10::scalar_value_type<scalar_t>::type delta
) {
  using std::exp;
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  scalar_t a_perturbed = a + scalar_t(delta);
  scalar_t z_inv = scalar_t(1) / z;

  scalar_t Pa = compute_neg_z_power(z, a_perturbed);
  scalar_t Pb = compute_neg_z_power(z, b);

  scalar_t dlogPa_da = compute_neg_z_power_log_derivative(z, a_perturbed);
  scalar_t dlogPb_db = compute_neg_z_power_log_derivative(z, b);

  scalar_t psi_c = digamma(c);
  scalar_t psi_b_minus_a = digamma(b - a_perturbed);
  scalar_t psi_a_minus_b = digamma(a_perturbed - b);
  scalar_t psi_a = digamma(a_perturbed);
  scalar_t psi_b = digamma(b);
  scalar_t psi_c_minus_a = digamma(c - a_perturbed);
  scalar_t psi_c_minus_b = digamma(c - b);

  scalar_t G1, G2;
  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_G1 = log_gamma_complex(c) + log_gamma_complex(b - a_perturbed)
                    - log_gamma_complex(b) - log_gamma_complex(c - a_perturbed);
    scalar_t log_G2 = log_gamma_complex(c) + log_gamma_complex(a_perturbed - b)
                    - log_gamma_complex(a_perturbed) - log_gamma_complex(c - b);
    G1 = exp(log_G1);
    G2 = exp(log_G2);
  } else {
    using std::lgamma;

    scalar_t log_mag_G1 = lgamma(c) + lgamma(b - a_perturbed) - lgamma(b) - lgamma(c - a_perturbed);
    scalar_t log_mag_G2 = lgamma(c) + lgamma(a_perturbed - b) - lgamma(a_perturbed) - lgamma(c - b);

    int sign_G1 = sign_gamma(c) * sign_gamma(b - a_perturbed)
                * sign_gamma(b) * sign_gamma(c - a_perturbed);
    int sign_G2 = sign_gamma(c) * sign_gamma(a_perturbed - b)
                * sign_gamma(a_perturbed) * sign_gamma(c - b);

    G1 = scalar_t(sign_G1) * exp(log_mag_G1);
    G2 = scalar_t(sign_G2) * exp(log_mag_G2);
  }

  scalar_t dlogG1_da = -psi_b_minus_a + psi_c_minus_a;
  scalar_t dlogG1_db = psi_b_minus_a - psi_b;
  scalar_t dlogG1_dc = psi_c - psi_c_minus_a;

  scalar_t dlogG2_da = psi_a_minus_b - psi_a;
  scalar_t dlogG2_db = -psi_a_minus_b + psi_c_minus_b;
  scalar_t dlogG2_dc = psi_c - psi_c_minus_b;

  scalar_t a1 = a_perturbed;
  scalar_t b1 = a_perturbed - c + scalar_t(1);
  scalar_t c1 = a_perturbed - b + scalar_t(1);
  auto [F1, dF1_da1, dF1_db1, dF1_dc1] =
      hypergeometric_2f1_series_with_param_derivatives(a1, b1, c1, z_inv);

  scalar_t dF1_da = dF1_da1 + dF1_db1 + dF1_dc1;
  scalar_t dF1_db = -dF1_dc1;
  scalar_t dF1_dc = -dF1_db1;

  scalar_t a2 = b;
  scalar_t b2 = b - c + scalar_t(1);
  scalar_t c2 = b - a_perturbed + scalar_t(1);
  auto [F2, dF2_da2, dF2_db2, dF2_dc2] =
      hypergeometric_2f1_series_with_param_derivatives(a2, b2, c2, z_inv);

  scalar_t dF2_da = -dF2_dc2;
  scalar_t dF2_db = dF2_da2 + dF2_db2 + dF2_dc2;
  scalar_t dF2_dc = -dF2_db2;

  scalar_t term1 = G1 * Pa * F1;
  scalar_t term2 = G2 * Pb * F2;
  scalar_t f = term1 + term2;

  scalar_t df_da = term1 * dlogG1_da + term1 * dlogPa_da + G1 * Pa * dF1_da
                 + term2 * dlogG2_da + G2 * Pb * dF2_da;

  scalar_t df_db = term1 * dlogG1_db + G1 * Pa * dF1_db
                 + term2 * dlogG2_db + term2 * dlogPb_db + G2 * Pb * dF2_db;

  scalar_t df_dc = term1 * dlogG1_dc + G1 * Pa * dF1_dc
                 + term2 * dlogG2_dc + G2 * Pb * dF2_dc;

  return std::make_tuple(f, df_da, df_db, df_dc);
}

/**
 * Explicit limiting form of linear transformation when a-b is a non-negative integer n.
 * Uses DLMF 15.8.8 formula instead of Richardson extrapolation for higher precision.
 *
 * For b = a - n (equivalently a - b = n >= 0), |z| > 1:
 *
 * ₂F₁(a, b; c; z) = [Γ(c)/Γ(c-b)] × (-z)^{-b} × {
 *   Σ_{k=0}^{n-1} [(b)_k (b-c+1)_k / k!] × (n-1-k)! × (1/z)^k
 *   + [(-1)^n / (n-1)!] × Σ_{s=0}^∞ [(a)_s (a-c+1)_s / (s!(s+n)!)] × (1/z)^{n+s}
 *     × [ψ(s+1) + ψ(s+n+1) - ψ(a+s) - ψ(a-c+1+s) - ln(-z)]
 * }
 *
 * For n = 0 (a = b), the polynomial sum is empty and the formula simplifies.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_linear_transform_integer_diff_explicit(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  int n
) {
  using std::exp;
  using std::log;
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  const int max_terms = 200;
  const real_t eps = std::numeric_limits<real_t>::epsilon() * real_t(100);

  // Check if a-c+1 is a non-positive integer (causes digamma pole in the series)
  // In this case, fall back to Richardson extrapolation with two delta values
  auto [ac1_is_int, ac1_int] = is_near_integer(a - c + scalar_t(1));
  if (ac1_is_int && ac1_int <= 0) {
    real_t delta1, delta2;
    if constexpr (std::is_same_v<real_t, double>) {
      delta1 = real_t(1e-7);
      delta2 = real_t(5e-8);
    } else {
      delta1 = real_t(1e-4);
      delta2 = real_t(5e-5);
    }
    scalar_t f1 = hypergeometric_2f1_linear_transform_with_delta(a, b, c, z, delta1);
    scalar_t f2 = hypergeometric_2f1_linear_transform_with_delta(a, b, c, z, delta2);
    return (f1 * scalar_t(delta2) - f2 * scalar_t(delta1)) / scalar_t(delta2 - delta1);
  }

  // Check if b-c+1 is a non-positive integer (causes issues in polynomial part)
  if (n >= 1) {
    auto [bc1_is_int, bc1_int] = is_near_integer(b - c + scalar_t(1));
    if (bc1_is_int && bc1_int <= 0) {
      real_t delta1, delta2;
      if constexpr (std::is_same_v<real_t, double>) {
        delta1 = real_t(1e-7);
        delta2 = real_t(5e-8);
      } else {
        delta1 = real_t(1e-4);
        delta2 = real_t(5e-5);
      }
      scalar_t f1 = hypergeometric_2f1_linear_transform_with_delta(a, b, c, z, delta1);
      scalar_t f2 = hypergeometric_2f1_linear_transform_with_delta(a, b, c, z, delta2);
      return (f1 * scalar_t(delta2) - f2 * scalar_t(delta1)) / scalar_t(delta2 - delta1);
    }
  }

  // Compute (-z)^{-b} and ln(-z)
  scalar_t neg_z = -z;
  scalar_t log_neg_z;
  scalar_t power_factor;

  if constexpr (c10::is_complex<scalar_t>::value) {
    log_neg_z = log(neg_z);
    power_factor = exp(-b * log_neg_z);
  } else {
    real_t z_val = static_cast<real_t>(z);
    if (z_val < real_t(0)) {
      // -z > 0, so log(-z) is real
      log_neg_z = log(-z_val);
      power_factor = exp(-b * log_neg_z);
    } else {
      // z > 0, -z < 0: log(-z) = log|z| + i*pi
      // (-z)^{-b} = |z|^{-b} * exp(-i*b*pi) = |z|^{-b} * (cos(b*pi) - i*sin(b*pi))
      // Return real part
      real_t log_abs_z = log(z_val);
      real_t pi_val = real_t(kPi_2F1);
      log_neg_z = log_abs_z;  // Store real part for the digamma terms
      power_factor = exp(-b * log_abs_z) * cos(b * pi_val);
    }
  }

  // Compute Gamma ratio: Γ(c) / Γ(c-b)
  scalar_t gamma_ratio_coeff;
  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_ratio = log_gamma_complex(c) - log_gamma_complex(c - b);
    gamma_ratio_coeff = exp(log_ratio);
  } else {
    using std::lgamma;
    scalar_t log_mag = lgamma(c) - lgamma(c - b);
    int sign = sign_gamma(c) * sign_gamma(c - b);
    gamma_ratio_coeff = scalar_t(sign) * exp(log_mag);
  }

  scalar_t z_inv = scalar_t(1) / z;
  scalar_t result = scalar_t(0);

  // =========================================================================
  // Part 1: Finite polynomial sum (only for n >= 1)
  // Σ_{k=0}^{n-1} [(b)_k (b-c+1)_k / k!] × (n-1-k)! × (1/z)^k
  // =========================================================================
  if (n >= 1) {
    scalar_t poly_sum = scalar_t(0);
    scalar_t poch_b = scalar_t(1);       // (b)_k
    scalar_t poch_bc1 = scalar_t(1);     // (b-c+1)_k
    scalar_t z_inv_pow = scalar_t(1);    // (1/z)^k
    scalar_t factorial_k = scalar_t(1);  // k!

    for (int k = 0; k < n; ++k) {
      // (n-1-k)! = Gamma(n-k)
      real_t factorial_nmk = real_t(1);
      for (int j = 2; j <= n - 1 - k; ++j) {
        factorial_nmk *= real_t(j);
      }

      scalar_t term = (poch_b * poch_bc1 / factorial_k) * scalar_t(factorial_nmk) * z_inv_pow;
      poly_sum += term;

      // Update for next iteration
      if (k < n - 1) {
        poch_b *= (b + scalar_t(k));
        poch_bc1 *= (b - c + scalar_t(1 + k));
        z_inv_pow *= z_inv;
        factorial_k *= scalar_t(k + 1);
      }
    }

    result += poly_sum;
  }

  // =========================================================================
  // Part 2: Infinite series with logarithmic terms
  // [(-1)^n / (n-1)!] × Σ_{s=0}^∞ [(a)_s (a-c+1)_s / (s!(s+n)!)] × (1/z)^{n+s}
  //   × [ψ(s+1) + ψ(s+n+1) - ψ(a+s) - ψ(a-c+1+s) - ln(-z)]
  // =========================================================================
  {
    // Compute (-1)^n / (n-1)! for n >= 1, or handle n = 0 specially
    scalar_t series_coeff;
    if (n == 0) {
      // For n = 0, the formula reduces to a simpler form with coefficient 1
      // and the digamma combination is [2ψ(s+1) - ψ(a+s) - ψ(a-c+1+s) - ln(-z)]
      series_coeff = scalar_t(1);
    } else {
      // (-1)^n / (n-1)!
      real_t factorial_nm1 = real_t(1);
      for (int j = 2; j <= n - 1; ++j) {
        factorial_nm1 *= real_t(j);
      }
      series_coeff = scalar_t((n % 2 == 0) ? 1 : -1) / scalar_t(factorial_nm1);
    }

    scalar_t log_series = scalar_t(0);
    scalar_t poch_a = scalar_t(1);       // (a)_s - need to advance to (a)_0 first
    scalar_t poch_ac1 = scalar_t(1);     // (a-c+1)_s
    scalar_t z_inv_pow = scalar_t(1);    // (1/z)^s

    // Precompute (1/z)^n
    scalar_t z_inv_n = scalar_t(1);
    for (int j = 0; j < n; ++j) {
      z_inv_n *= z_inv;
    }

    // Accumulate digamma sums for incremental computation
    // ψ(a+s) = ψ(a) + Σ_{j=0}^{s-1} 1/(a+j)
    scalar_t psi_sum_a = scalar_t(0);      // Σ 1/(a+j) from j=0 to s-1
    scalar_t psi_sum_ac1 = scalar_t(0);    // Σ 1/(a-c+1+j) from j=0 to s-1

    // Base digamma values
    scalar_t psi_a = digamma(a);
    scalar_t psi_ac1 = digamma(a - c + scalar_t(1));

    for (int s = 0; s < max_terms; ++s) {
      // Compute factorials: s! and (s+n)!
      real_t factorial_s = real_t(1);
      for (int j = 2; j <= s; ++j) {
        factorial_s *= real_t(j);
      }
      real_t factorial_sn = factorial_s;
      for (int j = s + 1; j <= s + n; ++j) {
        factorial_sn *= real_t(j);
      }

      // Digamma combination
      scalar_t psi_s1 = digamma(scalar_t(s + 1));        // ψ(s+1)
      scalar_t psi_sn1 = digamma(scalar_t(s + n + 1));   // ψ(s+n+1)
      scalar_t psi_as = psi_a + psi_sum_a;               // ψ(a+s)
      scalar_t psi_ac1s = psi_ac1 + psi_sum_ac1;         // ψ(a-c+1+s)

      scalar_t H;
      if (n == 0) {
        // For n = 0: [2ψ(s+1) - ψ(a+s) - ψ(a-c+1+s) - ln(-z)]
        H = scalar_t(2) * psi_s1 - psi_as - psi_ac1s - log_neg_z;
      } else {
        // For n >= 1: [ψ(s+1) + ψ(s+n+1) - ψ(a+s) - ψ(a-c+1+s) - ln(-z)]
        H = psi_s1 + psi_sn1 - psi_as - psi_ac1s - log_neg_z;
      }

      scalar_t term = (poch_a * poch_ac1) / scalar_t(factorial_s * factorial_sn) * z_inv_pow * z_inv_n * H;
      log_series += term;

      // Check convergence
      real_t term_mag = abs(term);
      real_t sum_mag = abs(log_series);
      if (term_mag < eps * (sum_mag > real_t(1) ? sum_mag : real_t(1))) {
        break;
      }

      // Update Pochhammer symbols and power for next iteration
      psi_sum_a += scalar_t(1) / (a + scalar_t(s));
      psi_sum_ac1 += scalar_t(1) / (a - c + scalar_t(1 + s));
      poch_a *= (a + scalar_t(s));
      poch_ac1 *= (a - c + scalar_t(1 + s));
      z_inv_pow *= z_inv;
    }

    result += series_coeff * log_series;
  }

  return gamma_ratio_coeff * power_factor * result;
}

/**
 * Limiting form of linear transformation when a-b is an integer n.
 * Dispatches to explicit formula for better precision.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t hypergeometric_2f1_linear_transform_integer_diff(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  int n
) {
  // Use symmetry for negative n: ₂F₁(a,b;c;z) = ₂F₁(b,a;c;z)
  if (n < 0) {
    return hypergeometric_2f1_linear_transform_integer_diff(b, a, c, z, -n);
  }

  return hypergeometric_2f1_linear_transform_integer_diff_explicit(a, b, c, z, n);
}

/**
 * Explicit derivative of linear transformation when a-b is a non-negative integer n.
 *
 * Computes the value and derivatives with respect to a, b, c analytically
 * using the DLMF 15.8.8 formula structure.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
hypergeometric_2f1_linear_transform_integer_diff_explicit_with_derivatives(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  int n
) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::cos;
  using std::sin;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  const int max_terms = 200;
  const real_t eps = std::numeric_limits<real_t>::epsilon() * real_t(100);

  // Check if a-c+1 is a non-positive integer (causes digamma pole in the series)
  // In this case, fall back to Richardson extrapolation with derivatives
  auto [ac1_is_int, ac1_int] = is_near_integer(a - c + scalar_t(1));
  if (ac1_is_int && ac1_int <= 0) {
    return hypergeometric_2f1_linear_transform_with_delta_and_derivatives(a, b, c, z, real_t(1e-8));
  }

  // Check if b-c+1 is a non-positive integer (causes issues in polynomial part)
  if (n >= 1) {
    auto [bc1_is_int, bc1_int] = is_near_integer(b - c + scalar_t(1));
    if (bc1_is_int && bc1_int <= 0) {
      return hypergeometric_2f1_linear_transform_with_delta_and_derivatives(a, b, c, z, real_t(1e-8));
    }
  }

  // Compute (-z)^{-b}, ln(-z), and their derivatives
  scalar_t neg_z = -z;
  scalar_t log_neg_z;
  scalar_t power_factor;
  scalar_t dlog_power_db;  // d/db of log((-z)^{-b}) = -ln(-z)

  if constexpr (c10::is_complex<scalar_t>::value) {
    log_neg_z = log(neg_z);
    power_factor = exp(-b * log_neg_z);
    dlog_power_db = -log_neg_z;
  } else {
    real_t z_val = static_cast<real_t>(z);
    if (z_val < real_t(0)) {
      log_neg_z = log(-z_val);
      power_factor = exp(-b * log_neg_z);
      dlog_power_db = -log_neg_z;
    } else {
      real_t log_abs_z = log(z_val);
      real_t pi_val = real_t(kPi_2F1);
      log_neg_z = log_abs_z;
      real_t b_val = static_cast<real_t>(b);
      power_factor = exp(-b * log_abs_z) * cos(b * pi_val);
      // d/db of |z|^{-b} cos(b*pi) = |z|^{-b} (-log|z| cos(b*pi) - pi sin(b*pi))
      dlog_power_db = -log_abs_z - pi_val * sin(b_val * pi_val) / cos(b_val * pi_val);
    }
  }

  // Gamma ratio: G = Γ(c) / Γ(c-b)
  // dlogG/db = ψ(c-b), dlogG/dc = ψ(c) - ψ(c-b)
  scalar_t psi_c = digamma(c);
  scalar_t psi_cmb = digamma(c - b);

  scalar_t gamma_ratio;
  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_ratio = log_gamma_complex(c) - log_gamma_complex(c - b);
    gamma_ratio = exp(log_ratio);
  } else {
    using std::lgamma;
    scalar_t log_mag = lgamma(c) - lgamma(c - b);
    int sign = sign_gamma(c) * sign_gamma(c - b);
    gamma_ratio = scalar_t(sign) * exp(log_mag);
  }

  scalar_t dlogG_da = scalar_t(0);  // G doesn't depend on a directly
  scalar_t dlogG_db = psi_cmb;
  scalar_t dlogG_dc = psi_c - psi_cmb;

  scalar_t z_inv = scalar_t(1) / z;

  // Initialize sums for value and derivatives
  scalar_t S = scalar_t(0);
  scalar_t dS_da = scalar_t(0);
  scalar_t dS_db = scalar_t(0);
  scalar_t dS_dc = scalar_t(0);

  // =========================================================================
  // Part 1: Finite polynomial sum (only for n >= 1)
  // P = Σ_{k=0}^{n-1} [(b)_k (b-c+1)_k / k!] × (n-1-k)! × (1/z)^k
  // =========================================================================
  if (n >= 1) {
    scalar_t poch_b = scalar_t(1);
    scalar_t poch_bc1 = scalar_t(1);
    scalar_t z_inv_pow = scalar_t(1);
    scalar_t factorial_k = scalar_t(1);
    scalar_t psi_sum_b = scalar_t(0);
    scalar_t psi_sum_bc1 = scalar_t(0);

    for (int k = 0; k < n; ++k) {
      real_t factorial_nmk = real_t(1);
      for (int j = 2; j <= n - 1 - k; ++j) {
        factorial_nmk *= real_t(j);
      }

      scalar_t base_term = (poch_b * poch_bc1 / factorial_k) * scalar_t(factorial_nmk) * z_inv_pow;
      S += base_term;

      // Derivatives with respect to b and c
      // d[(b)_k]/db = (b)_k × psi_sum_b
      // d[(b-c+1)_k]/db = (b-c+1)_k × psi_sum_bc1
      // d[(b-c+1)_k]/dc = -(b-c+1)_k × psi_sum_bc1
      if (k > 0) {
        dS_db += base_term * (psi_sum_b + psi_sum_bc1);
        dS_dc += base_term * (-psi_sum_bc1);
      }

      // Update for next iteration
      if (k < n - 1) {
        psi_sum_b += scalar_t(1) / (b + scalar_t(k));
        psi_sum_bc1 += scalar_t(1) / (b - c + scalar_t(1 + k));
        poch_b *= (b + scalar_t(k));
        poch_bc1 *= (b - c + scalar_t(1 + k));
        z_inv_pow *= z_inv;
        factorial_k *= scalar_t(k + 1);
      }
    }
  }

  // =========================================================================
  // Part 2: Infinite series with logarithmic terms
  // L = coeff × (1/z)^n × Σ_{s=0}^∞ [(a)_s (a-c+1)_s / (s!(s+n)!)] × (1/z)^s × H_s
  // where H_s = ψ(s+1) + ψ(s+n+1) - ψ(a+s) - ψ(a-c+1+s) - ln(-z)
  // =========================================================================
  {
    scalar_t series_coeff;
    if (n == 0) {
      series_coeff = scalar_t(1);
    } else {
      real_t factorial_nm1 = real_t(1);
      for (int j = 2; j <= n - 1; ++j) {
        factorial_nm1 *= real_t(j);
      }
      series_coeff = scalar_t((n % 2 == 0) ? 1 : -1) / scalar_t(factorial_nm1);
    }

    scalar_t z_inv_n = scalar_t(1);
    for (int j = 0; j < n; ++j) {
      z_inv_n *= z_inv;
    }

    scalar_t L = scalar_t(0);
    scalar_t dL_da = scalar_t(0);
    scalar_t dL_dc = scalar_t(0);

    scalar_t poch_a = scalar_t(1);
    scalar_t poch_ac1 = scalar_t(1);
    scalar_t z_inv_pow = scalar_t(1);
    scalar_t psi_sum_a = scalar_t(0);
    scalar_t psi_sum_ac1 = scalar_t(0);

    scalar_t psi_a_base = digamma(a);
    scalar_t psi_ac1_base = digamma(a - c + scalar_t(1));

    for (int s = 0; s < max_terms; ++s) {
      real_t factorial_s = real_t(1);
      for (int j = 2; j <= s; ++j) {
        factorial_s *= real_t(j);
      }
      real_t factorial_sn = factorial_s;
      for (int j = s + 1; j <= s + n; ++j) {
        factorial_sn *= real_t(j);
      }

      scalar_t psi_s1 = digamma(scalar_t(s + 1));
      scalar_t psi_sn1 = digamma(scalar_t(s + n + 1));
      scalar_t psi_as = psi_a_base + psi_sum_a;
      scalar_t psi_ac1s = psi_ac1_base + psi_sum_ac1;

      scalar_t H;
      if (n == 0) {
        H = scalar_t(2) * psi_s1 - psi_as - psi_ac1s - log_neg_z;
      } else {
        H = psi_s1 + psi_sn1 - psi_as - psi_ac1s - log_neg_z;
      }

      scalar_t coeff_term = (poch_a * poch_ac1) / scalar_t(factorial_s * factorial_sn) * z_inv_pow * z_inv_n;
      scalar_t term = coeff_term * H;
      L += term;

      // Derivatives
      scalar_t trigamma_as = scalar_t(0);
      scalar_t trigamma_ac1s = scalar_t(0);
      for (int j = 0; j <= s; ++j) {
        trigamma_as += scalar_t(1) / ((a + scalar_t(j)) * (a + scalar_t(j)));
        trigamma_ac1s += scalar_t(1) / ((a - c + scalar_t(1 + j)) * (a - c + scalar_t(1 + j)));
      }

      scalar_t dH_da = -trigamma_as - trigamma_ac1s;
      scalar_t dH_dc = trigamma_ac1s;  // d/dc of -ψ(a-c+1+s) = ψ'(a-c+1+s)

      // d(poch_a)/da = poch_a × psi_sum_a
      // d(poch_ac1)/da = poch_ac1 × psi_sum_ac1
      // d(poch_ac1)/dc = -poch_ac1 × psi_sum_ac1
      dL_da += coeff_term * (H * (psi_sum_a + psi_sum_ac1) + dH_da);
      dL_dc += coeff_term * (H * (-psi_sum_ac1) + dH_dc);

      // Convergence check
      real_t term_mag = abs(term);
      real_t sum_mag = abs(L);
      if (term_mag < eps * (sum_mag > real_t(1) ? sum_mag : real_t(1))) {
        break;
      }

      // Update for next iteration
      psi_sum_a += scalar_t(1) / (a + scalar_t(s));
      psi_sum_ac1 += scalar_t(1) / (a - c + scalar_t(1 + s));
      poch_a *= (a + scalar_t(s));
      poch_ac1 *= (a - c + scalar_t(1 + s));
      z_inv_pow *= z_inv;
    }

    S += series_coeff * L;
    dS_da += series_coeff * dL_da;
    dS_dc += series_coeff * dL_dc;
  }

  // Final result: f = G × P × S
  // where G = Γ(c)/Γ(c-b), P = (-z)^{-b}
  // df/da = G × P × dS/da
  // df/db = G × P × (S × (dlogG/db + dlogP/db) + dS/db)
  // df/dc = G × P × (S × dlogG/dc + dS/dc)
  scalar_t f = gamma_ratio * power_factor * S;
  scalar_t df_da = gamma_ratio * power_factor * dS_da;
  scalar_t df_db = gamma_ratio * power_factor * (S * (dlogG_db + dlog_power_db) + dS_db);
  scalar_t df_dc = gamma_ratio * power_factor * (S * dlogG_dc + dS_dc);

  return std::make_tuple(f, df_da, df_db, df_dc);
}

/**
 * Limiting form of linear transformation with derivatives.
 * Dispatches to explicit formula for better precision.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
hypergeometric_2f1_linear_transform_integer_diff_with_derivatives(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z,
  int n
) {
  if (n < 0) {
    auto [f, db, da, dc] = hypergeometric_2f1_linear_transform_integer_diff_with_derivatives(b, a, c, z, -n);
    return std::make_tuple(f, da, db, dc);
  }

  return hypergeometric_2f1_linear_transform_integer_diff_explicit_with_derivatives(a, b, c, z, n);
}

/**
 * Linear transformation for 2F1 when |z| > 1.
 *
 * For z < 0 with |z| > 1: Uses Pfaff transformation (DLMF 15.8.7):
 *   F(a, b; c; z) = (1-z)^{-a} × F(a, c-b; c; z/(z-1))
 *   which maps z to w = z/(z-1) with |w| < 1.
 *
 * For z > 1 or complex z: Uses DLMF 15.8.2 transformation with 1/z.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_linear_transform(scalar_t a, scalar_t b, scalar_t c, scalar_t z) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::cos;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  // For real z < 0 with |z| > 1, use Pfaff transformation (DLMF 15.8.7):
  // F(a, b; c; z) = (1-z)^{-a} × F(a, c-b; c; z/(z-1))
  // This maps z to w = z/(z-1) which satisfies |w| < 1 when z < -1.
  if constexpr (!c10::is_complex<scalar_t>::value) {
    if (z < real_t(0)) {
      scalar_t one_minus_z = scalar_t(1) - z;  // > 0 since z < 0
      scalar_t w = z / (z - scalar_t(1));  // w = z/(z-1) maps to (0, 1) for z < -1

      // Power factor: (1-z)^{-a}
      scalar_t power_factor = exp(-a * log(one_minus_z));

      // Transformed parameters: F(a, c-b; c; w)
      scalar_t b_new = c - b;

      // Compute transformed hypergeometric function recursively
      // Note: |w| < 1, so we can use series or 1-z transform
      return power_factor * hypergeometric_2_f_1(a, b_new, c, w);
    }
  }

  auto [is_int, n] = is_near_integer(a - b);
  if (is_int) {
    return hypergeometric_2f1_linear_transform_integer_diff(a, b, c, z, n);
  }

  scalar_t z_inv = scalar_t(1) / z;

  scalar_t factor_a = compute_neg_z_power(z, a);
  scalar_t factor_b = compute_neg_z_power(z, b);

  scalar_t G1, G2;
  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_G1 = log_gamma_complex(c) + log_gamma_complex(b - a)
                    - log_gamma_complex(b) - log_gamma_complex(c - a);
    scalar_t log_G2 = log_gamma_complex(c) + log_gamma_complex(a - b)
                    - log_gamma_complex(a) - log_gamma_complex(c - b);
    G1 = exp(log_G1);
    G2 = exp(log_G2);
  } else {
    using std::lgamma;

    scalar_t log_mag_G1 = lgamma(c) + lgamma(b - a) - lgamma(b) - lgamma(c - a);
    scalar_t log_mag_G2 = lgamma(c) + lgamma(a - b) - lgamma(a) - lgamma(c - b);

    int sign_G1 = sign_gamma(c) * sign_gamma(b - a) * sign_gamma(b) * sign_gamma(c - a);
    int sign_G2 = sign_gamma(c) * sign_gamma(a - b) * sign_gamma(a) * sign_gamma(c - b);

    G1 = scalar_t(sign_G1) * exp(log_mag_G1);
    G2 = scalar_t(sign_G2) * exp(log_mag_G2);
  }

  return G1 * factor_a * hypergeometric_2f1_series(a, a - c + scalar_t(1), a - b + scalar_t(1), z_inv)
       + G2 * factor_b * hypergeometric_2f1_series(b, b - c + scalar_t(1), b - a + scalar_t(1), z_inv);
}

/**
 * Linear transformation with analytical parameter derivatives.
 *
 * For z < 0 with |z| > 1: Uses Pfaff transformation (DLMF 15.8.7):
 *   F(a, b; c; z) = (1-z)^{-a} × F(a, c-b; c; z/(z-1))
 *
 * For z > 1 or complex z: Uses DLMF 15.8.2 transformation.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>
hypergeometric_2f1_linear_transform_with_param_derivatives(
    scalar_t a, scalar_t b, scalar_t c, scalar_t z
) {
  using std::exp;
  using std::log;
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  // For real z < 0 with |z| > 1, use Pfaff transformation (DLMF 15.8.7):
  // F(a, b; c; z) = (1-z)^{-a} × F(a, c-b; c; z/(z-1))
  if constexpr (!c10::is_complex<scalar_t>::value) {
    if (z < real_t(0)) {
      scalar_t one_minus_z = scalar_t(1) - z;
      scalar_t w = z / (z - scalar_t(1));

      // Power factor: P = (1-z)^{-a}
      scalar_t log_omz = log(one_minus_z);
      scalar_t P = exp(-a * log_omz);

      // Transformed parameters: F(a, c-b; c; w)
      scalar_t b_new = c - b;

      // Get analytical derivatives from inner function F(a, b_new; c; w)
      // where w is in (0, 1) since z < -1 implies w = z/(z-1) in (0, 1)
      // Use the appropriate derivative function based on w
      real_t w_mag = abs(w);
      scalar_t one_minus_w = scalar_t(1) - w;
      real_t one_minus_w_mag = abs(one_minus_w);

      scalar_t F_inner, dF_inner_da, dF_inner_db_new, dF_inner_dc;
      if (one_minus_w_mag < w_mag) {
        // Use 1-z transform for w near 1
        auto [f_inner, da_inner, db_inner, dc_inner] =
            hypergeometric_2f1_one_minus_z_transform_with_param_derivatives(a, b_new, c, w);
        F_inner = f_inner;
        dF_inner_da = da_inner;
        dF_inner_db_new = db_inner;
        dF_inner_dc = dc_inner;
      } else {
        // Use series for w near 0
        auto [f_inner, da_inner, db_inner, dc_inner] =
            hypergeometric_2f1_series_with_param_derivatives(a, b_new, c, w);
        F_inner = f_inner;
        dF_inner_da = da_inner;
        dF_inner_db_new = db_inner;
        dF_inner_dc = dc_inner;
      }

      // Final result: f = P × F_inner
      scalar_t f = P * F_inner;

      // Chain rule for derivatives:
      // df/da = dP/da × F_inner + P × dF_inner/da
      //       = (-ln(1-z)) × P × F_inner + P × dF_inner/da
      scalar_t df_da = (-log_omz) * P * F_inner + P * dF_inner_da;

      // df/db: b_new = c - b, so d(b_new)/db = -1
      // df/db = P × dF_inner/db_new × (-1)
      scalar_t df_db = P * (-dF_inner_db_new);

      // df/dc: b_new = c - b, so d(b_new)/dc = 1
      // df/dc = P × (dF_inner/dc + dF_inner/db_new × 1)
      scalar_t df_dc = P * (dF_inner_dc + dF_inner_db_new);

      return std::make_tuple(f, df_da, df_db, df_dc);
    }
  }

  auto [is_int, n] = is_near_integer(a - b);
  if (is_int) {
    return hypergeometric_2f1_linear_transform_integer_diff_with_derivatives(a, b, c, z, n);
  }

  scalar_t z_inv = scalar_t(1) / z;

  scalar_t Pa = compute_neg_z_power(z, a);
  scalar_t Pb = compute_neg_z_power(z, b);

  scalar_t dlogPa_da = compute_neg_z_power_log_derivative(z, a);
  scalar_t dlogPb_db = compute_neg_z_power_log_derivative(z, b);

  scalar_t psi_c = digamma(c);
  scalar_t psi_b_minus_a = digamma(b - a);
  scalar_t psi_a_minus_b = digamma(a - b);
  scalar_t psi_a = digamma(a);
  scalar_t psi_b = digamma(b);
  scalar_t psi_c_minus_a = digamma(c - a);
  scalar_t psi_c_minus_b = digamma(c - b);

  scalar_t G1, G2;
  if constexpr (c10::is_complex<scalar_t>::value) {
    scalar_t log_G1 = log_gamma_complex(c) + log_gamma_complex(b - a)
                    - log_gamma_complex(b) - log_gamma_complex(c - a);
    scalar_t log_G2 = log_gamma_complex(c) + log_gamma_complex(a - b)
                    - log_gamma_complex(a) - log_gamma_complex(c - b);
    G1 = exp(log_G1);
    G2 = exp(log_G2);
  } else {
    using std::lgamma;

    scalar_t log_mag_G1 = lgamma(c) + lgamma(b - a) - lgamma(b) - lgamma(c - a);
    scalar_t log_mag_G2 = lgamma(c) + lgamma(a - b) - lgamma(a) - lgamma(c - b);

    int sign_G1 = sign_gamma(c) * sign_gamma(b - a) * sign_gamma(b) * sign_gamma(c - a);
    int sign_G2 = sign_gamma(c) * sign_gamma(a - b) * sign_gamma(a) * sign_gamma(c - b);

    G1 = scalar_t(sign_G1) * exp(log_mag_G1);
    G2 = scalar_t(sign_G2) * exp(log_mag_G2);
  }

  scalar_t dlogG1_da = -psi_b_minus_a + psi_c_minus_a;
  scalar_t dlogG1_db = psi_b_minus_a - psi_b;
  scalar_t dlogG1_dc = psi_c - psi_c_minus_a;

  scalar_t dlogG2_da = psi_a_minus_b - psi_a;
  scalar_t dlogG2_db = -psi_a_minus_b + psi_c_minus_b;
  scalar_t dlogG2_dc = psi_c - psi_c_minus_b;

  scalar_t a1 = a;
  scalar_t b1 = a - c + scalar_t(1);
  scalar_t c1 = a - b + scalar_t(1);
  auto [F1, dF1_da1, dF1_db1, dF1_dc1] =
      hypergeometric_2f1_series_with_param_derivatives(a1, b1, c1, z_inv);

  scalar_t dF1_da = dF1_da1 + dF1_db1 + dF1_dc1;
  scalar_t dF1_db = -dF1_dc1;
  scalar_t dF1_dc = -dF1_db1;

  scalar_t a2 = b;
  scalar_t b2 = b - c + scalar_t(1);
  scalar_t c2 = b - a + scalar_t(1);
  auto [F2, dF2_da2, dF2_db2, dF2_dc2] =
      hypergeometric_2f1_series_with_param_derivatives(a2, b2, c2, z_inv);

  scalar_t dF2_da = -dF2_dc2;
  scalar_t dF2_db = dF2_da2 + dF2_db2 + dF2_dc2;
  scalar_t dF2_dc = -dF2_db2;

  scalar_t term1 = G1 * Pa * F1;
  scalar_t term2 = G2 * Pb * F2;
  scalar_t f = term1 + term2;

  scalar_t df_da = term1 * dlogG1_da + term1 * dlogPa_da + G1 * Pa * dF1_da
                 + term2 * dlogG2_da + G2 * Pb * dF2_da;

  scalar_t df_db = term1 * dlogG1_db + G1 * Pa * dF1_db
                 + term2 * dlogG2_db + term2 * dlogPb_db + G2 * Pb * dF2_db;

  scalar_t df_dc = term1 * dlogG1_dc + G1 * Pa * dF1_dc
                 + term2 * dlogG2_dc + G2 * Pb * dF2_dc;

  return std::make_tuple(f, df_da, df_db, df_dc);
}

// ============================================================================
// Main function
// ============================================================================

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t hypergeometric_2_f_1(
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z
) {
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  real_t z_mag = abs(z);

  if (z_mag >= real_t(1)) {
    return hypergeometric_2f1_linear_transform(a, b, c, z);
  }

  scalar_t one_minus_z = scalar_t(1) - z;
  real_t one_minus_z_mag = abs(one_minus_z);

  if (one_minus_z_mag < z_mag) {
    return hypergeometric_2f1_one_minus_z_transform(a, b, c, z);
  }

  return hypergeometric_2f1_series(a, b, c, z);
}

// ============================================================================
// Backward pass
// ============================================================================

/**
 * Backward pass for 2F1: computes gradients w.r.t. a, b, c, z.
 *
 * Uses analytical gradients for all algorithm branches, including
 * integer parameter difference cases via Richardson extrapolation.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> hypergeometric_2_f_1_backward(
  scalar_t gradient_output,
  scalar_t a,
  scalar_t b,
  scalar_t c,
  scalar_t z
) {
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  real_t z_mag = abs(z);
  scalar_t one_minus_z = scalar_t(1) - z;
  real_t one_minus_z_mag = abs(one_minus_z);

  scalar_t d_a, d_b, d_c, d_z;

  if (z_mag >= real_t(1)) {
    auto [value, da, db, dc] = hypergeometric_2f1_linear_transform_with_param_derivatives(a, b, c, z);
    (void)value;

    d_a = da;
    d_b = db;
    d_c = dc;
    d_z = hypergeometric_2f1_derivative(a, b, c, z);
  } else if (one_minus_z_mag < z_mag) {
    auto [value, da, db, dc] = hypergeometric_2f1_one_minus_z_transform_with_param_derivatives(a, b, c, z);
    (void)value;

    d_a = da;
    d_b = db;
    d_c = dc;
    d_z = hypergeometric_2f1_derivative(a, b, c, z);
  } else {
    auto [value, da, db, dc] = hypergeometric_2f1_series_with_param_derivatives(a, b, c, z);
    (void)value;

    d_a = da;
    d_b = db;
    d_c = dc;
    d_z = hypergeometric_2f1_derivative(a, b, c, z);
  }

  scalar_t gradient_a, gradient_b, gradient_c, gradient_z;
  if constexpr (c10::is_complex<scalar_t>::value) {
    gradient_a = gradient_output * std::conj(d_a);
    gradient_b = gradient_output * std::conj(d_b);
    gradient_c = gradient_output * std::conj(d_c);
    gradient_z = gradient_output * std::conj(d_z);
  } else {
    gradient_a = gradient_output * d_a;
    gradient_b = gradient_output * d_b;
    gradient_c = gradient_output * d_c;
    gradient_z = gradient_output * d_z;
  }

  return std::make_tuple(gradient_a, gradient_b, gradient_c, gradient_z);
}

/**
 * Second-order backward pass for 2F1.
 *
 * Computes gradients of the first-order gradients with respect to all inputs.
 * Uses finite differences of the analytical first derivatives to compute
 * second-order mixed partial derivatives.
 *
 * The first backward pass computes:
 *   grad_a = gradient_output * df/da
 *   grad_b = gradient_output * df/db
 *   grad_c = gradient_output * df/dc
 *   grad_z = gradient_output * df/dz
 *
 * This second backward pass computes gradients of these w.r.t. all inputs:
 *   gradient_gradient_output = sum_x(gradient_gradient_x * df/dx)
 *   gradient_a = gradient_output * sum_x(gradient_gradient_x * d²f/dxda)
 *   gradient_b = gradient_output * sum_x(gradient_gradient_x * d²f/dxdb)
 *   gradient_c = gradient_output * sum_x(gradient_gradient_x * d²f/dxdc)
 *   gradient_z = gradient_output * sum_x(gradient_gradient_x * d²f/dxdz)
 *
 * Second derivatives are computed via central finite differences of the
 * analytical first derivatives, except d²f/dz² which uses the analytical
 * formula: d²f/dz² = (ab/c) * ((a+1)(b+1)/(c+1)) * 2F1(a+2,b+2;c+2;z)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t> hypergeometric_2_f_1_backward_backward(
    scalar_t gradient_gradient_a,
    scalar_t gradient_gradient_b,
    scalar_t gradient_gradient_c,
    scalar_t gradient_gradient_z,
    scalar_t gradient_output,
    scalar_t a,
    scalar_t b,
    scalar_t c,
    scalar_t z,
    const bool has_gradient_gradient_a,
    const bool has_gradient_gradient_b,
    const bool has_gradient_gradient_c,
    const bool has_gradient_gradient_z
) {
  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  if (!has_gradient_gradient_a && !has_gradient_gradient_b && !has_gradient_gradient_c && !has_gradient_gradient_z) {
    return std::make_tuple(
      scalar_t(0),
      scalar_t(0),
      scalar_t(0),
      scalar_t(0),
      scalar_t(0)
    );
  }

  real_t h;
  if constexpr (std::is_same_v<real_t, double>) {
    h = real_t(1e-7);
  } else {
    h = real_t(1e-4);
  }

  scalar_t h_s = scalar_t(h);
  scalar_t two_h = scalar_t(2 * h);

  // Get analytical first derivatives at the base point
  // hypergeometric_2_f_1_backward returns (grad_a, grad_b, grad_c, grad_z)
  // where grad_x = gradient_output * df/dx
  // Passing gradient_output=1 gives us the raw derivatives
  auto [df_da, df_db, df_dc, df_dz] = hypergeometric_2_f_1_backward(scalar_t(1), a, b, c, z);

  // Initialize outputs
  scalar_t gradient_gradient_output = scalar_t(0);
  scalar_t gradient_a = scalar_t(0);
  scalar_t gradient_b = scalar_t(0);
  scalar_t gradient_c = scalar_t(0);
  scalar_t gradient_z = scalar_t(0);

  // =========================================================================
  // Compute gradient_gradient_output using analytical first derivatives
  // gradient_gradient_output = sum_x(gradient_gradient_x * df/dx)
  // =========================================================================
  if constexpr (c10::is_complex<scalar_t>::value) {
    if (has_gradient_gradient_a) gradient_gradient_output += gradient_gradient_a * std::conj(df_da);
    if (has_gradient_gradient_b) gradient_gradient_output += gradient_gradient_b * std::conj(df_db);
    if (has_gradient_gradient_c) gradient_gradient_output += gradient_gradient_c * std::conj(df_dc);
    if (has_gradient_gradient_z) gradient_gradient_output += gradient_gradient_z * std::conj(df_dz);
  } else {
    if (has_gradient_gradient_a) gradient_gradient_output += gradient_gradient_a * df_da;
    if (has_gradient_gradient_b) gradient_gradient_output += gradient_gradient_b * df_db;
    if (has_gradient_gradient_c) gradient_gradient_output += gradient_gradient_c * df_dc;
    if (has_gradient_gradient_z) gradient_gradient_output += gradient_gradient_z * df_dz;
  }

  // =========================================================================
  // Compute second derivatives via finite differences of first derivatives
  // We compute derivatives at perturbed points and use central differences
  // =========================================================================

  // Get first derivatives at perturbed a values
  auto [df_da_ap, df_db_ap, df_dc_ap, df_dz_ap] = hypergeometric_2_f_1_backward(scalar_t(1), a + h_s, b, c, z);
  auto [df_da_am, df_db_am, df_dc_am, df_dz_am] = hypergeometric_2_f_1_backward(scalar_t(1), a - h_s, b, c, z);

  // Get first derivatives at perturbed b values
  auto [df_da_bp, df_db_bp, df_dc_bp, df_dz_bp] = hypergeometric_2_f_1_backward(scalar_t(1), a, b + h_s, c, z);
  auto [df_da_bm, df_db_bm, df_dc_bm, df_dz_bm] = hypergeometric_2_f_1_backward(scalar_t(1), a, b - h_s, c, z);

  // Get first derivatives at perturbed c values
  auto [df_da_cp, df_db_cp, df_dc_cp, df_dz_cp] = hypergeometric_2_f_1_backward(scalar_t(1), a, b, c + h_s, z);
  auto [df_da_cm, df_db_cm, df_dc_cm, df_dz_cm] = hypergeometric_2_f_1_backward(scalar_t(1), a, b, c - h_s, z);

  // Get first derivatives at perturbed z values
  auto [df_da_zp, df_db_zp, df_dc_zp, df_dz_zp] = hypergeometric_2_f_1_backward(scalar_t(1), a, b, c, z + h_s);
  auto [df_da_zm, df_db_zm, df_dc_zm, df_dz_zm] = hypergeometric_2_f_1_backward(scalar_t(1), a, b, c, z - h_s);

  // Compute pure second derivatives: d²f/dx²
  scalar_t d2f_da2 = (df_da_ap - df_da_am) / two_h;
  scalar_t d2f_db2 = (df_db_bp - df_db_bm) / two_h;
  scalar_t d2f_dc2 = (df_dc_cp - df_dc_cm) / two_h;

  // For d²f/dz², use analytical formula for better accuracy:
  // d²f/dz² = (ab/c) * ((a+1)(b+1)/(c+1)) * 2F1(a+2, b+2; c+2; z)
  scalar_t d2f_dz2 = (a * b / c) * ((a + scalar_t(1)) * (b + scalar_t(1)) / (c + scalar_t(1)))
                   * hypergeometric_2_f_1(a + scalar_t(2), b + scalar_t(2), c + scalar_t(2), z);

  // Compute mixed second derivatives: d²f/dxdy (symmetric, so d²f/dxdy = d²f/dydx)
  // d²f/dadb = d(df/da)/db = (df/da(b+h) - df/da(b-h)) / (2h)
  scalar_t d2f_dadb = (df_da_bp - df_da_bm) / two_h;
  scalar_t d2f_dadc = (df_da_cp - df_da_cm) / two_h;
  scalar_t d2f_dadz = (df_da_zp - df_da_zm) / two_h;
  scalar_t d2f_dbdc = (df_db_cp - df_db_cm) / two_h;
  scalar_t d2f_dbdz = (df_db_zp - df_db_zm) / two_h;
  scalar_t d2f_dcdz = (df_dc_zp - df_dc_zm) / two_h;

  // =========================================================================
  // Compute gradient contributions from each gradient_gradient_x
  // gradient_y = gradient_output * sum_x(gradient_gradient_x * d²f/dxdy)
  // =========================================================================
  if constexpr (c10::is_complex<scalar_t>::value) {
    if (has_gradient_gradient_a) {
      gradient_a += gradient_gradient_a * gradient_output * std::conj(d2f_da2);
      gradient_b += gradient_gradient_a * gradient_output * std::conj(d2f_dadb);
      gradient_c += gradient_gradient_a * gradient_output * std::conj(d2f_dadc);
      gradient_z += gradient_gradient_a * gradient_output * std::conj(d2f_dadz);
    }
    if (has_gradient_gradient_b) {
      gradient_a += gradient_gradient_b * gradient_output * std::conj(d2f_dadb);  // symmetric
      gradient_b += gradient_gradient_b * gradient_output * std::conj(d2f_db2);
      gradient_c += gradient_gradient_b * gradient_output * std::conj(d2f_dbdc);
      gradient_z += gradient_gradient_b * gradient_output * std::conj(d2f_dbdz);
    }
    if (has_gradient_gradient_c) {
      gradient_a += gradient_gradient_c * gradient_output * std::conj(d2f_dadc);  // symmetric
      gradient_b += gradient_gradient_c * gradient_output * std::conj(d2f_dbdc);  // symmetric
      gradient_c += gradient_gradient_c * gradient_output * std::conj(d2f_dc2);
      gradient_z += gradient_gradient_c * gradient_output * std::conj(d2f_dcdz);
    }
    if (has_gradient_gradient_z) {
      gradient_a += gradient_gradient_z * gradient_output * std::conj(d2f_dadz);  // symmetric
      gradient_b += gradient_gradient_z * gradient_output * std::conj(d2f_dbdz);  // symmetric
      gradient_c += gradient_gradient_z * gradient_output * std::conj(d2f_dcdz);  // symmetric
      gradient_z += gradient_gradient_z * gradient_output * std::conj(d2f_dz2);
    }
  } else {
    if (has_gradient_gradient_a) {
      gradient_a += gradient_gradient_a * gradient_output * d2f_da2;
      gradient_b += gradient_gradient_a * gradient_output * d2f_dadb;
      gradient_c += gradient_gradient_a * gradient_output * d2f_dadc;
      gradient_z += gradient_gradient_a * gradient_output * d2f_dadz;
    }
    if (has_gradient_gradient_b) {
      gradient_a += gradient_gradient_b * gradient_output * d2f_dadb;  // symmetric
      gradient_b += gradient_gradient_b * gradient_output * d2f_db2;
      gradient_c += gradient_gradient_b * gradient_output * d2f_dbdc;
      gradient_z += gradient_gradient_b * gradient_output * d2f_dbdz;
    }
    if (has_gradient_gradient_c) {
      gradient_a += gradient_gradient_c * gradient_output * d2f_dadc;  // symmetric
      gradient_b += gradient_gradient_c * gradient_output * d2f_dbdc;  // symmetric
      gradient_c += gradient_gradient_c * gradient_output * d2f_dc2;
      gradient_z += gradient_gradient_c * gradient_output * d2f_dcdz;
    }
    if (has_gradient_gradient_z) {
      gradient_a += gradient_gradient_z * gradient_output * d2f_dadz;  // symmetric
      gradient_b += gradient_gradient_z * gradient_output * d2f_dbdz;  // symmetric
      gradient_c += gradient_gradient_z * gradient_output * d2f_dcdz;  // symmetric
      gradient_z += gradient_gradient_z * gradient_output * d2f_dz2;
    }
  }

  return std::make_tuple(
    gradient_gradient_output,
    gradient_a,
    gradient_b,
    gradient_c,
    gradient_z
  );
}

}  // namespace torchscience::impl::special_functions
