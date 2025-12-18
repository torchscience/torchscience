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
 * 3. ANALYTIC CONTINUATION:
 *    For |z| > 1, uses the linear transformation (DLMF 15.8.2):
 *
 *    2F1(a,b;c;z) = G1 * (-z)^{-a} * 2F1(a, a-c+1; a-b+1; 1/z)
 *                + G2 * (-z)^{-b} * 2F1(b, b-c+1; b-a+1; 1/z)
 *
 *    where:
 *    G1 = Gamma(c) * Gamma(b-a) / (Gamma(b) * Gamma(c-a))
 *    G2 = Gamma(c) * Gamma(a-b) / (Gamma(a) * Gamma(c-b))
 *
 * 4. SPECIAL HANDLING:
 *    When a-b is an integer, the standard formula has poles. A regularization
 *    approach is used to handle this limiting case.
 *
 * 5. APPLICATIONS:
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
#include "digamma.h"  // for is_nonpositive_integer
#include "sin_pi.h"

namespace torchscience::impl::special_functions {

// Pi constant (duplicated from gamma.h for local use)
constexpr double kPi_2F1 = 3.14159265358979323846264338327950288;

// Euler-Mascheroni constant for limiting form calculations
constexpr double kEulerMascheroni = 0.5772156649015328606065120900824024;

/**
 * Log of the gamma function for complex arguments: log(Gamma(z))
 *
 * Uses Lanczos approximation in log form to avoid overflow/underflow:
 *   log(Gamma(z)) = 0.5*log(2*pi) + (z-0.5)*log(t) - t + log(A_g(z))
 *   where t = z + g - 0.5 and A_g is the Lanczos series.
 *
 * For Re(z) < 0.5, uses the reflection formula:
 *   log(Gamma(z)) = log(pi) - log(sin(pi*z)) - log(Gamma(1-z))
 *
 * Returns infinity at poles (non-positive integers).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> log_gamma_complex(c10::complex<T> z) {
  using std::log;
  using std::abs;

  const T pi_val = T(kPi_2F1);
  const T g = T(kLanczosG);

  // Helper to create real-valued complex constants
  const auto real = [](T val) { return c10::complex<T>(val, T(0)); };

  // Check for poles at non-positive integers
  if (is_nonpositive_integer(z)) {
    return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
  }

  // For Re(z) < 0.5, use reflection formula:
  // log(Gamma(z)) = log(pi) - log(sin(pi*z)) - log(Gamma(1-z))
  if (z.real() < T(0.5)) {
    auto sin_pi_z = sin_pi(z);
    // Handle case where sin(pi*z) is zero (shouldn't happen if not at pole)
    if (abs(sin_pi_z) < T(1e-300)) {
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }
    return real(log(pi_val)) - log(sin_pi_z) -
           log_gamma_complex(real(T(1)) - z);
  }

  // Lanczos approximation in log form for Re(z) >= 0.5
  // log(Gamma(z)) = 0.5*log(2*pi) + (z-0.5)*log(t) - t + log(A_g(z))
  // where t = z + g - 0.5
  auto A_g = lanczos_series(z);  // From gamma.h
  auto t = z + real(g - T(0.5));

  // Compute in log form: 0.5*log(2*pi) + (z-0.5)*log(t) - t + log(A_g)
  return real(T(0.5) * log(T(2) * pi_val)) +
         (z - real(T(0.5))) * log(t) - t + log(A_g);
}

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
 * Special care is taken to avoid overflow/underflow by computing terms
 * incrementally using the ratio of successive terms.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_series(scalar_t a, scalar_t b, scalar_t c, scalar_t z) {
  using std::abs;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  const int max_terms = 200;
  const real_t eps = std::numeric_limits<real_t>::epsilon() * real_t(100);

  // Handle z = 0 case
  real_t z_mag;
  if constexpr (c10::is_complex<scalar_t>::value) {
    z_mag = abs(z);
  } else {
    z_mag = abs(z);
  }
  if (z_mag < eps) {
    return scalar_t(1);
  }

  // Series: 2F1 = 1 + (ab/c)*z + (a(a+1)*b(b+1))/(c(c+1)*2!)*z^2 + ...
  // Compute iteratively: term_{n+1} = term_n * (a+n)*(b+n)/((c+n)*(n+1)) * z
  scalar_t sum = scalar_t(1);
  scalar_t term = scalar_t(1);

  for (int n = 0; n < max_terms; ++n) {
    // Compute ratio: (a+n)*(b+n) / ((c+n)*(n+1))
    scalar_t a_n = a + scalar_t(n);
    scalar_t b_n = b + scalar_t(n);
    scalar_t c_n = c + scalar_t(n);
    scalar_t n_plus_1 = scalar_t(n + 1);

    // Check for c being a non-positive integer (pole)
    real_t c_n_mag;
    if constexpr (c10::is_complex<scalar_t>::value) {
      c_n_mag = abs(c_n);
    } else {
      c_n_mag = abs(c_n);
    }
    if (c_n_mag < eps) {
      // c + n is zero - pole in the series
      return scalar_t(std::numeric_limits<real_t>::quiet_NaN());
    }

    term *= (a_n * b_n) / (c_n * n_plus_1) * z;
    sum += term;

    // Check for convergence
    real_t term_mag, sum_mag;
    if constexpr (c10::is_complex<scalar_t>::value) {
      term_mag = abs(term);
      sum_mag = abs(sum);
    } else {
      term_mag = abs(term);
      sum_mag = abs(sum);
    }

    if (term_mag < eps * sum_mag) {
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
  scalar_t hyp = hypergeometric_2f1_series(a + scalar_t(1), b + scalar_t(1),
                                           c + scalar_t(1), z);
  return coeff * hyp;
}

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

  // Dtype-aware tolerance
  real_t tol;
  if constexpr (std::is_same_v<real_t, double>) {
    tol = real_t(1e-10);
  } else {
    tol = real_t(1e-5);
  }

  real_t x_real;
  if constexpr (c10::is_complex<scalar_t>::value) {
    // For complex, check if imaginary part is small and real part is near integer
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

// Forward declaration for mutual recursion
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_linear_transform_integer_diff(
    scalar_t a, scalar_t b, scalar_t c, scalar_t z, int n);

/**
 * Linear transformation for 2F1 when |z| > 1 (DLMF 15.8.2).
 *
 * For |z| > 1, uses the formula:
 *   2F1(a,b;c;z) = G1 * (-z)^{-a} * 2F1(a, a-c+1; a-b+1; 1/z)
 *               + G2 * (-z)^{-b} * 2F1(b, b-c+1; b-a+1; 1/z)
 *
 * where:
 *   G1 = Gamma(c) * Gamma(b-a) / (Gamma(b) * Gamma(c-a))
 *   G2 = Gamma(c) * Gamma(a-b) / (Gamma(a) * Gamma(c-b))
 *
 * This is valid when a-b is not an integer. For integer a-b, use the
 * limiting form hypergeometric_2f1_linear_transform_integer_diff().
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_linear_transform(scalar_t a, scalar_t b, scalar_t c, scalar_t z) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::cos;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  // Check if a-b is near an integer - if so, use limiting form
  auto [is_int, n] = is_near_integer(a - b);
  if (is_int) {
    return hypergeometric_2f1_linear_transform_integer_diff(a, b, c, z, n);
  }

  // Compute 1/z for the transformed 2F1 arguments
  scalar_t z_inv = scalar_t(1) / z;

  // Compute (-z)^{-a} and (-z)^{-b} using principal branch
  // (-z)^{-a} = exp(-a * log(-z))
  // For principal branch: log(-z) = log|z| + i*(arg(z) + pi) for Im(z) >= 0
  //                                = log|z| + i*(arg(z) - pi) for Im(z) < 0
  scalar_t neg_z = -z;
  scalar_t log_neg_z;
  if constexpr (c10::is_complex<scalar_t>::value) {
    log_neg_z = log(neg_z);
  } else {
    // For real z > 0: -z < 0, so log(-z) = log|z| + i*pi
    // But since we're in real mode, this means z is on the branch cut
    // We should compute this as complex
    log_neg_z = log(abs(neg_z));  // Real part only - imaginary part handled below
  }

  scalar_t factor_a = exp(-a * log_neg_z);
  scalar_t factor_b = exp(-b * log_neg_z);

  // For real z > 1, we need to account for the i*pi in log(-z)
  // (-z)^{-a} = |z|^{-a} * exp(-a * i * pi) = |z|^{-a} * (cos(a*pi) - i*sin(a*pi))
  // But since result should be real for real inputs, we need careful handling
  if constexpr (!c10::is_complex<scalar_t>::value) {
    // For real z > 1, the result is actually complex in general
    // However, for the incomplete beta with real a, b, the imaginary parts cancel
    // We compute using the real-valued version that works when a, b cause cancellation
    real_t pi_val = real_t(kPi_2F1);

    // |z|^{-a} * cos(a*pi) and |z|^{-b} * cos(b*pi)
    real_t z_abs = abs(z);
    real_t pow_a = exp(-a * log(z_abs));
    real_t pow_b = exp(-b * log(z_abs));

    factor_a = pow_a * cos(a * pi_val);
    factor_b = pow_b * cos(b * pi_val);
  }

  // Gamma ratios using log_gamma for numerical stability
  // G1 = Gamma(c) * Gamma(b-a) / (Gamma(b) * Gamma(c-a))
  // G2 = Gamma(c) * Gamma(a-b) / (Gamma(a) * Gamma(c-b))
  scalar_t log_G1, log_G2;
  if constexpr (c10::is_complex<scalar_t>::value) {
    log_G1 = log_gamma_complex(c) + log_gamma_complex(b - a)
           - log_gamma_complex(b) - log_gamma_complex(c - a);
    log_G2 = log_gamma_complex(c) + log_gamma_complex(a - b)
           - log_gamma_complex(a) - log_gamma_complex(c - b);
  } else {
    using std::lgamma;
    log_G1 = lgamma(c) + lgamma(b - a) - lgamma(b) - lgamma(c - a);
    log_G2 = lgamma(c) + lgamma(a - b) - lgamma(a) - lgamma(c - b);
  }

  scalar_t G1 = exp(log_G1);
  scalar_t G2 = exp(log_G2);

  // Compute the two 2F1 terms at 1/z (which has |1/z| < 1 since |z| > 1)
  // Term 1: 2F1(a, a-c+1; a-b+1; 1/z)
  scalar_t hyp_a = a;
  scalar_t hyp_b1 = a - c + scalar_t(1);
  scalar_t hyp_c1 = a - b + scalar_t(1);
  scalar_t hyp_1 = hypergeometric_2f1_series(hyp_a, hyp_b1, hyp_c1, z_inv);

  // Term 2: 2F1(b, b-c+1; b-a+1; 1/z)
  scalar_t hyp_b2 = b - c + scalar_t(1);
  scalar_t hyp_c2 = b - a + scalar_t(1);
  scalar_t hyp_2 = hypergeometric_2f1_series(b, hyp_b2, hyp_c2, z_inv);

  return G1 * factor_a * hyp_1 + G2 * factor_b * hyp_2;
}

/**
 * Limiting form of linear transformation when a-b is an integer n.
 *
 * When a - b = n (integer), the standard formula has poles in Gamma(b-a) and
 * Gamma(a-b). The limiting form involves logarithmic terms (DLMF 15.8.10).
 *
 * For n >= 0 (a >= b):
 *   The formula involves a single 2F1 term plus logarithmic corrections
 *   with digamma (psi) functions.
 *
 * For n < 0 (a < b):
 *   Use symmetry to reduce to n >= 0 case.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
hypergeometric_2f1_linear_transform_integer_diff(
    scalar_t a, scalar_t b, scalar_t c, scalar_t z, int n
) {
  using std::exp;
  using std::log;
  using std::abs;
  using std::cos;

  using real_t = typename c10::scalar_value_type<scalar_t>::type;

  // For n < 0, use the symmetry 2F1(a,b;c;z) = 2F1(b,a;c;z) to get n >= 0
  if (n < 0) {
    return hypergeometric_2f1_linear_transform_integer_diff(b, a, c, z, -n);
  }

  // Now n >= 0, so a = b + n
  // The limiting form (DLMF 15.8.10) for a - b = n (non-negative integer):
  //
  // 2F1(a,b;c;z) = [Gamma(c) / Gamma(a)] * sum_{k=0}^{n-1} (a-n)_k * (1-c+a-n)_k
  //                * (k+n-1)! / (k! * (n-k-1)!) * (-z)^{-a+k}
  //              + [Gamma(c) * (-1)^n / (Gamma(a) * Gamma(c-a) * n!)]
  //                * (-z)^{-a} * sum_{k=0}^{inf} (a)_k * (1-c+a)_k / (k! * (k+n)!)
  //                * (1/z)^k * [log(-z) + psi(k+1) + psi(k+n+1) - psi(a+k) - psi(1-c+a+k)]
  //
  // This is quite complex. For simplicity, we use a regularization approach:
  // perturb a slightly and use the standard formula, which is accurate for most cases.

  // Regularization: perturb a-b slightly away from integer
  real_t delta = real_t(1e-8);
  scalar_t a_perturbed = a + scalar_t(delta);

  // Recompute using standard formula with perturbed parameters
  scalar_t z_inv = scalar_t(1) / z;
  scalar_t neg_z = -z;
  scalar_t log_neg_z;
  if constexpr (c10::is_complex<scalar_t>::value) {
    log_neg_z = log(neg_z);
  } else {
    log_neg_z = log(abs(neg_z));
  }

  scalar_t factor_a = exp(-a_perturbed * log_neg_z);
  scalar_t factor_b = exp(-b * log_neg_z);

  if constexpr (!c10::is_complex<scalar_t>::value) {
    real_t pi_val = real_t(kPi_2F1);
    real_t z_abs = abs(z);
    factor_a = exp(-a_perturbed * log(z_abs)) * cos(a_perturbed * pi_val);
    factor_b = exp(-b * log(z_abs)) * cos(b * pi_val);
  }

  // Gamma ratios with perturbed a
  scalar_t log_G1, log_G2;
  if constexpr (c10::is_complex<scalar_t>::value) {
    log_G1 = log_gamma_complex(c) + log_gamma_complex(b - a_perturbed)
           - log_gamma_complex(b) - log_gamma_complex(c - a_perturbed);
    log_G2 = log_gamma_complex(c) + log_gamma_complex(a_perturbed - b)
           - log_gamma_complex(a_perturbed) - log_gamma_complex(c - b);
  } else {
    using std::lgamma;
    log_G1 = lgamma(c) + lgamma(b - a_perturbed) - lgamma(b) - lgamma(c - a_perturbed);
    log_G2 = lgamma(c) + lgamma(a_perturbed - b) - lgamma(a_perturbed) - lgamma(c - b);
  }

  scalar_t G1 = exp(log_G1);
  scalar_t G2 = exp(log_G2);

  // 2F1 terms with perturbed parameters
  scalar_t hyp_1 = hypergeometric_2f1_series(
      a_perturbed, a_perturbed - c + scalar_t(1),
      a_perturbed - b + scalar_t(1), z_inv);
  scalar_t hyp_2 = hypergeometric_2f1_series(
      b, b - c + scalar_t(1),
      b - a_perturbed + scalar_t(1), z_inv);

  return G1 * factor_a * hyp_1 + G2 * factor_b * hyp_2;
}

}  // namespace torchscience::impl::special_functions
