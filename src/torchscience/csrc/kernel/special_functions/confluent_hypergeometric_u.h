#pragma once

#include <cmath>
#include "cmath_compat.h"
#include <complex>
#include <limits>
#include <type_traits>

#include <c10/util/complex.h>

#include "confluent_hypergeometric_m.h"
#include "digamma.h"
#include "log_gamma.h"
#include "gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Type traits for confluent hypergeometric U (use hypu_ prefix to avoid ODR conflicts)
template <typename T>
struct hypu_is_complex_type : std::false_type {};

template <typename T>
struct hypu_is_complex_type<std::complex<T>> : std::true_type {};

template <typename T>
struct hypu_is_complex_type<c10::complex<T>> : std::true_type {};

template <typename T>
C10_HOST_DEVICE inline constexpr bool hypu_is_complex_v = hypu_is_complex_type<T>::value;

template <typename T>
struct hypu_real_type { using type = T; };

template <typename T>
struct hypu_real_type<std::complex<T>> { using type = T; };

template <typename T>
struct hypu_real_type<c10::complex<T>> { using type = T; };

template <typename T>
C10_HOST_DEVICE using hypu_real_type_t = typename hypu_real_type<T>::type;

template <typename T>
C10_HOST_DEVICE constexpr auto hypu_epsilon() {
  using real_t = hypu_real_type_t<T>;
  if constexpr (std::is_same_v<real_t, float>) {
    return float(1e-6);
  } else if constexpr (std::is_same_v<real_t, double>) {
    return double(1e-14);
  } else {
    return float(1e-6);
  }
}

template <typename T>
C10_HOST_DEVICE bool hypu_is_nonpositive_integer(T x) {
  if constexpr (hypu_is_complex_v<T>) {
    using real_t = hypu_real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < hypu_epsilon<T>() &&
           re <= real_t(0) &&
           std::abs(re - std::round(re)) < hypu_epsilon<T>();
  } else {
    double xd = static_cast<double>(x);
    return xd <= 0.0 && std::abs(xd - std::round(xd)) < hypu_epsilon<T>();
  }
}

template <typename T>
C10_HOST_DEVICE bool hypu_is_positive_integer(T x) {
  if constexpr (hypu_is_complex_v<T>) {
    using real_t = hypu_real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < hypu_epsilon<T>() &&
           re >= real_t(1) &&
           std::abs(re - std::round(re)) < hypu_epsilon<T>();
  } else {
    double xd = static_cast<double>(x);
    return xd >= 1.0 && std::abs(xd - std::round(xd)) < hypu_epsilon<T>();
  }
}

template <typename T>
C10_HOST_DEVICE bool hypu_is_integer(T x) {
  if constexpr (hypu_is_complex_v<T>) {
    using real_t = hypu_real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < hypu_epsilon<T>() &&
           std::abs(re - std::round(re)) < hypu_epsilon<T>();
  } else {
    double xd = static_cast<double>(x);
    return std::abs(xd - std::round(xd)) < hypu_epsilon<T>();
  }
}

template <typename T>
C10_HOST_DEVICE int hypu_get_integer(T x) {
  if constexpr (hypu_is_complex_v<T>) {
    using real_t = hypu_real_type_t<T>;
    return static_cast<int>(std::round(static_cast<real_t>(x.real())));
  } else {
    return static_cast<int>(std::round(static_cast<double>(x)));
  }
}

// Returns the sign of Gamma(x) for real x.
// For x > 0: Gamma(x) > 0, sign = +1.
// For x < 0 (non-integer): sign depends on floor(x).
//   Gamma(x) > 0 when floor(x) is even, < 0 when floor(x) is odd.
template <typename T>
C10_HOST_DEVICE int hypu_gamma_sign(T x) {
  if constexpr (hypu_is_complex_v<T>) {
    // For complex arguments, sign tracking via log_gamma phase is correct;
    // this helper is only meaningful for real arguments.
    return 1;
  } else {
    if (x > T(0)) {
      return 1;
    }
    // x < 0 and not a non-positive integer (caller should have checked)
    int fl = static_cast<int>(std::floor(static_cast<double>(x)));
    return (fl % 2 == 0) ? 1 : -1;
  }
}

// Asymptotic expansion for large |z|:
// U(a, b, z) ~ z^(-a) * sum_{n=0}^{inf} (a)_n * (a - b + 1)_n / n! * (-z)^(-n)
template <typename T>
C10_HOST_DEVICE T hypu_asymptotic(T a, T b, T z, int max_iter = 100) {
  T sum = T(1);
  T term = T(1);
  T a_minus_b_plus_1 = a - b + T(1);
  T neg_z_inv = T(-1) / z;

  for (int n = 0; n < max_iter; ++n) {
    T new_term = term * (a + T(n)) * (a_minus_b_plus_1 + T(n)) / T(n + 1) * neg_z_inv;

    // Check for convergence
    if (std::abs(new_term) < hypu_epsilon<T>() * std::abs(sum)) {
      break;
    }

    // Check for divergence (asymptotic series may diverge)
    if (std::abs(new_term) > std::abs(term) && n > 5) {
      break;
    }

    term = new_term;
    sum += term;
  }

  // Compute z^(-a)
  T result = std::exp(-a * std::log(z)) * sum;
  return result;
}

// U for integer b >= 1 using DLMF 13.2.10 logarithmic limiting form.
//
// U(a, n, z) = ((-1)^n / (Gamma(a-n+1) * (n-1)!)) *
//              [ M(a, n, z) * ln(z)
//                + sum_{k=0}^{inf} (a)_k / ((n)_k * k!) * z^k *
//                  (psi(a+k) - psi(1+k) - psi(n+k)) ]
//            + ((n-2)! / Gamma(a)) *
//              sum_{k=0}^{n-2} (a-n+1)_k / ((2-n)_k * k!) * z^(k+1-n)
template <typename T>
C10_HOST_DEVICE T hypu_integer_b_positive(T a, int n, T z, int max_iter = 200) {
  // --- First part: logarithmic series ---
  // Coefficient: (-1)^n / (Gamma(a - n + 1) * (n-1)!)
  T a_minus_n_plus_1 = a - T(n) + T(1);
  T log_gamma_a_n1 = log_gamma(a_minus_n_plus_1);

  T first_part = T(0);

  if (!cmath_compat::isinf(log_gamma_a_n1)) {
    // (n-1)! as T
    T factorial_n_minus_1 = T(1);
    for (int i = 2; i < n; ++i) {
      factorial_n_minus_1 *= T(i);
    }

    T sign_n = (n % 2 == 0) ? T(1) : T(-1);
    int gamma_sign = hypu_gamma_sign(a_minus_n_plus_1);
    T coeff = sign_n * T(gamma_sign) * std::exp(-log_gamma_a_n1) / factorial_n_minus_1;

    // M(a, n, z) * ln(z)
    T M_val = confluent_hypergeometric_m(a, T(n), z);
    T log_z = std::log(z);
    T log_term = M_val * log_z;

    // Infinite series: sum_{k=0}^{inf} (a)_k / ((n)_k * k!) * z^k *
    //                  (psi(a+k) - psi(1+k) - psi(n+k))
    T series_sum = T(0);
    T pochhammer_a = T(1);   // (a)_k, starts at 1 for k=0
    T pochhammer_n = T(1);   // (n)_k, starts at 1 for k=0
    T k_factorial = T(1);    // k!, starts at 1 for k=0
    T z_power = T(1);        // z^k, starts at 1 for k=0

    for (int k = 0; k < max_iter; ++k) {
      T psi_a_k = digamma(a + T(k));
      T psi_1_k = digamma(T(1 + k));
      T psi_n_k = digamma(T(n + k));

      T term_coeff = pochhammer_a / (pochhammer_n * k_factorial);
      T term = term_coeff * z_power * (psi_a_k - psi_1_k - psi_n_k);
      series_sum += term;

      // Check convergence
      if (k > 0 && std::abs(term) < hypu_epsilon<T>() * std::abs(series_sum)) {
        break;
      }

      // Update Pochhammer symbols and factorials for next iteration
      pochhammer_a *= (a + T(k));
      pochhammer_n *= T(n + k);
      k_factorial *= T(k + 1);
      z_power *= z;
    }

    first_part = coeff * (log_term + series_sum);
  }

  // --- Second part: finite sum (only exists for n >= 2) ---
  // ((n-2)! / Gamma(a)) * sum_{k=0}^{n-2} (a-n+1)_k / ((2-n)_k * k!) * z^(k+1-n)
  T second_part = T(0);

  if (n >= 2) {
    T log_gamma_a = log_gamma(a);

    if (!cmath_compat::isinf(log_gamma_a)) {
      T factorial_n_minus_2 = T(1);
      for (int i = 2; i <= n - 2; ++i) {
        factorial_n_minus_2 *= T(i);
      }

      int gamma_sign_a = hypu_gamma_sign(a);
      T outer_coeff = T(gamma_sign_a) * factorial_n_minus_2 * std::exp(-log_gamma_a);

      T finite_sum = T(0);
      T poch_a_n1 = T(1);   // (a-n+1)_k
      T poch_2_n = T(1);    // (2-n)_k
      T k_fact = T(1);      // k!
      T z_pow = std::exp(T(1 - n) * std::log(z));  // z^(1-n) for k=0

      for (int k = 0; k <= n - 2; ++k) {
        T term = poch_a_n1 / (poch_2_n * k_fact) * z_pow;
        finite_sum += term;

        // Update for next iteration (guard prevents computing the
        // zero Pochhammer factor (2-n)_{n-1} on the final iteration)
        if (k < n - 2) {
          poch_a_n1 *= (a - T(n) + T(1) + T(k));
          poch_2_n *= (T(2 - n) + T(k));
          k_fact *= T(k + 1);
          z_pow *= z;
        }
      }

      second_part = outer_coeff * finite_sum;
    }
  }

  return first_part + second_part;
}

// U for integer b <= 0 using limiting form
template <typename T>
C10_HOST_DEVICE T hypu_integer_b_nonpositive(T a, int n, T z) {
  // For b = n <= 0 (non-positive integer), the formula simplifies
  // Both 1-b and 2-b are positive integers > 1
  T b = T(n);
  T one_minus_b = T(1) - b;  // Positive integer >= 1
  T a_minus_b_plus_1 = a - b + T(1);

  // Gamma(1-b) = (|b|)! for b <= 0
  T log_gamma_1_minus_b = log_gamma(one_minus_b);
  T log_gamma_a_minus_b_plus_1 = log_gamma(a_minus_b_plus_1);

  if (cmath_compat::isinf(log_gamma_a_minus_b_plus_1)) {
    // a - b + 1 is a non-positive integer
    T b_minus_1 = b - T(1);  // Negative
    T log_gamma_b_minus_1 = log_gamma(b_minus_1);
    T log_gamma_a = log_gamma(a);

    if (cmath_compat::isinf(log_gamma_b_minus_1) || cmath_compat::isinf(log_gamma_a)) {
      return std::numeric_limits<T>::quiet_NaN();
    }

    T coeff2 = std::exp(log_gamma_b_minus_1 - log_gamma_a);
    T sign2 = T(hypu_gamma_sign(b_minus_1) * hypu_gamma_sign(a));
    T z_power = std::exp((T(1) - b) * std::log(z));
    T M2 = confluent_hypergeometric_m(a - b + T(1), T(2) - b, z);

    return sign2 * coeff2 * z_power * M2;
  }

  T coeff1 = std::exp(log_gamma_1_minus_b - log_gamma_a_minus_b_plus_1);
  T sign1 = T(hypu_gamma_sign(one_minus_b) * hypu_gamma_sign(a_minus_b_plus_1));
  T M1 = confluent_hypergeometric_m(a, b, z);

  // Second term
  T b_minus_1 = b - T(1);
  T log_gamma_b_minus_1 = log_gamma(b_minus_1);
  T log_gamma_a = log_gamma(a);

  if (cmath_compat::isinf(log_gamma_b_minus_1)) {
    // b - 1 is a non-positive integer, second term vanishes
    return sign1 * coeff1 * M1;
  }

  if (cmath_compat::isinf(log_gamma_a)) {
    // a is a non-positive integer, second term vanishes
    return sign1 * coeff1 * M1;
  }

  T coeff2 = std::exp(log_gamma_b_minus_1 - log_gamma_a);
  T sign2 = T(hypu_gamma_sign(b_minus_1) * hypu_gamma_sign(a));
  T z_power = std::exp((T(1) - b) * std::log(z));
  T M2 = confluent_hypergeometric_m(a - b + T(1), T(2) - b, z);

  return sign1 * coeff1 * M1 + sign2 * coeff2 * z_power * M2;
}

// General U computation using M-based definition for non-integer b
// U(a, b, z) = Gamma(1-b)/Gamma(a-b+1) * M(a, b, z)
//            + Gamma(b-1)/Gamma(a) * z^(1-b) * M(a-b+1, 2-b, z)
//
// Key improvement: compute both terms in log-space and combine using
// log-sum-exp style arithmetic to avoid catastrophic cancellation when
// the two terms have similar magnitude but opposite sign.
template <typename T>
C10_HOST_DEVICE T hypu_via_m(T a, T b, T z) {
  T one_minus_b = T(1) - b;
  T a_minus_b_plus_1 = a - b + T(1);

  // First term: Gamma(1-b) / Gamma(a-b+1) * M(a, b, z)
  T log_gamma_1_minus_b = log_gamma(one_minus_b);
  T log_gamma_a_minus_b_plus_1 = log_gamma(a_minus_b_plus_1);

  T log_abs_term1 = T(0);
  T sign1 = T(0);
  bool term1_valid = true;

  if (cmath_compat::isinf(log_gamma_1_minus_b) || cmath_compat::isinf(log_gamma_a_minus_b_plus_1)) {
    term1_valid = false;
  } else {
    T M1 = confluent_hypergeometric_m(a, b, z);
    T coeff1 = std::exp(log_gamma_1_minus_b - log_gamma_a_minus_b_plus_1);
    T gamma_ratio_sign1 = T(hypu_gamma_sign(one_minus_b) * hypu_gamma_sign(a_minus_b_plus_1));
    T val1 = gamma_ratio_sign1 * coeff1 * M1;
    log_abs_term1 = std::log(std::abs(val1));
    sign1 = val1 >= T(0) ? T(1) : T(-1);
  }

  // Second term: Gamma(b-1) / Gamma(a) * z^(1-b) * M(a-b+1, 2-b, z)
  T b_minus_1 = b - T(1);
  T log_gamma_b_minus_1 = log_gamma(b_minus_1);
  T log_gamma_a = log_gamma(a);

  T log_abs_term2 = T(0);
  T sign2 = T(0);
  bool term2_valid = true;

  if (cmath_compat::isinf(log_gamma_b_minus_1) || cmath_compat::isinf(log_gamma_a)) {
    term2_valid = false;
  } else {
    T coeff2 = std::exp(log_gamma_b_minus_1 - log_gamma_a);
    T gamma_ratio_sign2 = T(hypu_gamma_sign(b_minus_1) * hypu_gamma_sign(a));
    T z_power = std::exp(one_minus_b * std::log(z));
    T M2 = confluent_hypergeometric_m(a_minus_b_plus_1, T(2) - b, z);
    T val2 = gamma_ratio_sign2 * coeff2 * z_power * M2;
    log_abs_term2 = std::log(std::abs(val2));
    sign2 = val2 >= T(0) ? T(1) : T(-1);
  }

  if (!term1_valid && !term2_valid) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  if (!term1_valid) {
    return sign2 * std::exp(log_abs_term2);
  }
  if (!term2_valid) {
    return sign1 * std::exp(log_abs_term1);
  }

  // Log-sum-exp style combination to avoid catastrophic cancellation.
  // We have: result = sign1 * exp(log1) + sign2 * exp(log2)
  // Let M = max(log1, log2):
  //   result = exp(M) * (sign1 * exp(log1 - M) + sign2 * exp(log2 - M))
  // The subtraction (log_i - M) is <= 0, so exp() is in [0, 1] and well-conditioned.
  T log_max = std::max(log_abs_term1, log_abs_term2);

  // Guard: if both terms are zero, log_max is -inf and subtraction produces NaN.
  if (cmath_compat::isinf(log_max) && log_max < T(0)) {
    return T(0);
  }

  T combined = sign1 * std::exp(log_abs_term1 - log_max)
             + sign2 * std::exp(log_abs_term2 - log_max);
  return combined * std::exp(log_max);
}

// Complex U for integer b >= 1 using DLMF 13.2.10 logarithmic limiting form.
// Same formula as hypu_integer_b_positive but adapted for complex arithmetic
// where log_gamma gives the full complex value (no sign tracking needed).
template <typename T>
C10_HOST_DEVICE c10::complex<T> hypu_integer_b_positive_complex(c10::complex<T> a, int n, c10::complex<T> z, int max_iter = 200) {
  using C = c10::complex<T>;
  C one(T(1), T(0));
  C zero(T(0), T(0));

  // --- First part: logarithmic series ---
  // Coefficient: (-1)^n / (Gamma(a - n + 1) * (n-1)!)
  C a_minus_n_plus_1 = a - C(T(n), T(0)) + one;
  C log_gamma_a_n1 = log_gamma(a_minus_n_plus_1);

  C first_part = zero;

  if (!cmath_compat::isinf(log_gamma_a_n1.real())) {
    // (n-1)!
    T factorial_n_minus_1 = T(1);
    for (int i = 2; i < n; ++i) {
      factorial_n_minus_1 *= T(i);
    }

    C sign_n = (n % 2 == 0) ? one : C(T(-1), T(0));
    C coeff = sign_n / (std::exp(log_gamma_a_n1) * C(factorial_n_minus_1, T(0)));

    // M(a, n, z) * ln(z)
    C M_val = confluent_hypergeometric_m(a, C(T(n), T(0)), z);
    C log_z = std::log(z);
    C log_term = M_val * log_z;

    // Infinite series: sum_{k=0}^{inf} (a)_k / ((n)_k * k!) * z^k *
    //                  (psi(a+k) - psi(1+k) - psi(n+k))
    C series_sum = zero;
    C pochhammer_a = one;
    C pochhammer_n = one;
    T k_factorial = T(1);
    C z_power = one;

    for (int k = 0; k < max_iter; ++k) {
      C psi_a_k = digamma(a + C(T(k), T(0)));
      C psi_1_k = digamma(C(T(1 + k), T(0)));
      C psi_n_k = digamma(C(T(n + k), T(0)));

      C term_coeff = pochhammer_a / (pochhammer_n * C(k_factorial, T(0)));
      C term = term_coeff * z_power * (psi_a_k - psi_1_k - psi_n_k);
      series_sum = series_sum + term;

      if (k > 0 && std::abs(term) < hypu_epsilon<C>() * std::abs(series_sum)) {
        break;
      }

      pochhammer_a = pochhammer_a * (a + C(T(k), T(0)));
      pochhammer_n = pochhammer_n * C(T(n + k), T(0));
      k_factorial *= T(k + 1);
      z_power = z_power * z;
    }

    first_part = coeff * (log_term + series_sum);
  }

  // --- Second part: finite sum (only exists for n >= 2) ---
  C second_part = zero;

  if (n >= 2) {
    C log_gamma_a = log_gamma(a);

    if (!cmath_compat::isinf(log_gamma_a.real())) {
      T factorial_n_minus_2 = T(1);
      for (int i = 2; i <= n - 2; ++i) {
        factorial_n_minus_2 *= T(i);
      }

      C outer_coeff = C(factorial_n_minus_2, T(0)) / std::exp(log_gamma_a);

      C finite_sum = zero;
      C poch_a_n1 = one;
      C poch_2_n = one;
      T k_fact = T(1);
      C z_pow = std::exp(C(T(1 - n), T(0)) * std::log(z));

      for (int k = 0; k <= n - 2; ++k) {
        C term = poch_a_n1 / (poch_2_n * C(k_fact, T(0))) * z_pow;
        finite_sum = finite_sum + term;

        if (k < n - 2) {
          poch_a_n1 = poch_a_n1 * (a - C(T(n), T(0)) + one + C(T(k), T(0)));
          poch_2_n = poch_2_n * C(T(2 - n + k), T(0));
          k_fact *= T(k + 1);
          z_pow = z_pow * z;
        }
      }

      second_part = outer_coeff * finite_sum;
    }
  }

  return first_part + second_part;
}

} // namespace detail

// Confluent hypergeometric function U(a, b, z) (Tricomi function)
template <typename T>
C10_HOST_DEVICE T confluent_hypergeometric_u(T a, T b, T z) {
  using detail::hypu_epsilon;
  using detail::hypu_is_nonpositive_integer;
  using detail::hypu_is_positive_integer;
  using detail::hypu_is_integer;
  using detail::hypu_get_integer;
  using detail::hypu_is_complex_v;
  using detail::hypu_asymptotic;
  using detail::hypu_via_m;
  using detail::hypu_integer_b_positive;

  // Handle z = 0: U(a, b, 0) has a pole for Re(b) > 1
  // For Re(b) <= 1, U(a, b, 0) = Gamma(1-b) / Gamma(a-b+1)
  if (std::abs(z) < hypu_epsilon<T>()) {
    // Check if b > 1 (pole at z = 0)
    double b_real;
    if constexpr (hypu_is_complex_v<T>) {
      b_real = static_cast<double>(b.real());
    } else {
      b_real = static_cast<double>(b);
    }

    if (b_real > 1.0 + hypu_epsilon<T>()) {
      return std::numeric_limits<T>::infinity();
    }

    // U(a, b, 0) = Gamma(1-b) / Gamma(a-b+1) for Re(b) <= 1
    T one_minus_b = T(1) - b;
    T a_minus_b_plus_1 = a - b + T(1);

    T log_gamma_num = log_gamma(one_minus_b);
    T log_gamma_denom = log_gamma(a_minus_b_plus_1);

    if (cmath_compat::isinf(log_gamma_denom)) {
      // a - b + 1 is a non-positive integer
      return T(0);
    }
    if (cmath_compat::isinf(log_gamma_num)) {
      // 1 - b is a non-positive integer
      return std::numeric_limits<T>::infinity();
    }

    return std::exp(log_gamma_num - log_gamma_denom);
  }

  // Special case: a = 0, U(0, b, z) = 1
  if (std::abs(a) < hypu_epsilon<T>()) {
    return T(1);
  }

  // Special case: a = b, U(a, a, z) = exp(z) * Gamma(1-a, z) / z^(a-1)
  // For integer a >= 1: U(n, n, z) = exp(z) * sum_{k=0}^{n-1} z^k / k!
  // This is related to incomplete gamma, but for now use general method

  // Get magnitude of z
  double z_abs;
  if constexpr (hypu_is_complex_v<T>) {
    z_abs = std::abs(z);
  } else {
    z_abs = std::abs(static_cast<double>(z));
  }

  // For large |z|, use asymptotic expansion
  if (z_abs > 30.0) {
    return hypu_asymptotic(a, b, z);
  }

  // Check if b is an integer
  if (hypu_is_integer(b)) {
    int b_int = hypu_get_integer(b);

    if (b_int >= 2) {
      // b is a positive integer >= 2, need special handling
      return hypu_integer_b_positive(a, b_int, z);
    } else if (b_int == 1) {
      // b = 1: route through DLMF 13.2.10 (finite sum vanishes for n=1)
      return hypu_integer_b_positive(a, 1, z);
    }
    // b <= 0: standard formula works
  }

  // General case: use M-based formula
  return hypu_via_m(a, b, z);
}

// Complex version
template <typename T>
C10_HOST_DEVICE c10::complex<T> confluent_hypergeometric_u(c10::complex<T> a, c10::complex<T> b, c10::complex<T> z) {
  using detail::hypu_epsilon;
  using detail::hypu_is_nonpositive_integer;
  using detail::hypu_is_integer;
  using detail::hypu_get_integer;

  // Handle z = 0
  if (std::abs(z) < hypu_epsilon<c10::complex<T>>()) {
    // Check if Re(b) > 1 (pole at z = 0)
    if (b.real() > T(1) + hypu_epsilon<c10::complex<T>>()) {
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    // U(a, b, 0) = Gamma(1-b) / Gamma(a-b+1) for Re(b) <= 1
    c10::complex<T> one_minus_b = c10::complex<T>(T(1), T(0)) - b;
    c10::complex<T> a_minus_b_plus_1 = a - b + c10::complex<T>(T(1), T(0));

    c10::complex<T> log_gamma_num = log_gamma(one_minus_b);
    c10::complex<T> log_gamma_denom = log_gamma(a_minus_b_plus_1);

    if (cmath_compat::isinf(log_gamma_denom.real())) {
      return c10::complex<T>(T(0), T(0));
    }
    if (cmath_compat::isinf(log_gamma_num.real())) {
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    return std::exp(log_gamma_num - log_gamma_denom);
  }

  // Special case: a = 0
  if (std::abs(a) < hypu_epsilon<c10::complex<T>>()) {
    return c10::complex<T>(T(1), T(0));
  }

  double z_abs = std::abs(z);

  // For large |z|, use asymptotic expansion
  if (z_abs > 30.0) {
    c10::complex<T> sum(T(1), T(0));
    c10::complex<T> term(T(1), T(0));
    c10::complex<T> a_minus_b_plus_1 = a - b + c10::complex<T>(T(1), T(0));
    c10::complex<T> neg_z_inv = c10::complex<T>(T(-1), T(0)) / z;

    for (int n = 0; n < 100; ++n) {
      c10::complex<T> n_c(T(n), T(0));
      c10::complex<T> new_term = term * (a + n_c) * (a_minus_b_plus_1 + n_c) / c10::complex<T>(T(n + 1), T(0)) * neg_z_inv;

      if (std::abs(new_term) < hypu_epsilon<c10::complex<T>>() * std::abs(sum)) {
        break;
      }
      if (std::abs(new_term) > std::abs(term) && n > 5) {
        break;
      }

      term = new_term;
      sum += term;
    }

    // z^(-a) * sum
    return std::exp(-a * std::log(z)) * sum;
  }

  // Check if b is an integer (real part integer, imaginary part ~ 0)
  if (hypu_is_integer(b)) {
    int b_int = hypu_get_integer(b);

    if (b_int >= 1) {
      return detail::hypu_integer_b_positive_complex(a, b_int, z);
    }
    // b <= 0: general formula works
  }

  // General case: use M-based formula
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> one_minus_b = one - b;
  c10::complex<T> a_minus_b_plus_1 = a - b + one;

  // First term: Gamma(1-b) / Gamma(a-b+1) * M(a, b, z)
  c10::complex<T> log_gamma_1_minus_b = log_gamma(one_minus_b);
  c10::complex<T> log_gamma_a_minus_b_plus_1 = log_gamma(a_minus_b_plus_1);

  c10::complex<T> term1(T(0), T(0));
  bool term1_valid = true;

  if (cmath_compat::isinf(log_gamma_1_minus_b.real())) {
    term1_valid = false;
  } else if (cmath_compat::isinf(log_gamma_a_minus_b_plus_1.real())) {
    term1_valid = false;
  } else {
    c10::complex<T> coeff1 = std::exp(log_gamma_1_minus_b - log_gamma_a_minus_b_plus_1);
    c10::complex<T> M1 = confluent_hypergeometric_m(a, b, z);
    term1 = coeff1 * M1;
  }

  // Second term: Gamma(b-1) / Gamma(a) * z^(1-b) * M(a-b+1, 2-b, z)
  c10::complex<T> b_minus_1 = b - one;
  c10::complex<T> log_gamma_b_minus_1 = log_gamma(b_minus_1);
  c10::complex<T> log_gamma_a = log_gamma(a);

  c10::complex<T> term2(T(0), T(0));
  bool term2_valid = true;

  if (cmath_compat::isinf(log_gamma_b_minus_1.real())) {
    term2_valid = false;
  } else if (cmath_compat::isinf(log_gamma_a.real())) {
    term2_valid = false;
  } else {
    c10::complex<T> coeff2 = std::exp(log_gamma_b_minus_1 - log_gamma_a);
    c10::complex<T> z_power = std::exp(one_minus_b * std::log(z));
    c10::complex<T> two_minus_b = c10::complex<T>(T(2), T(0)) - b;
    c10::complex<T> M2 = confluent_hypergeometric_m(a_minus_b_plus_1, two_minus_b, z);
    term2 = coeff2 * z_power * M2;
  }

  if (!term1_valid && !term2_valid) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), T(0));
  }

  return (term1_valid ? term1 : c10::complex<T>(T(0), T(0))) +
         (term2_valid ? term2 : c10::complex<T>(T(0), T(0)));
}

} // namespace torchscience::kernel::special_functions
