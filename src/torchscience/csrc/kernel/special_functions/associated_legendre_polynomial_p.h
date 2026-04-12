#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "cmath_compat.h"
#include <limits>

namespace torchscience::kernel::special_functions {

// Associated Legendre polynomial P_n^m(x)
//
// Computes P_n^m(x) via three-term recurrence with Condon-Shortley phase.
//
// Mathematical Definition:
//   P_n^m(x) = (-1)^m (1-x^2)^{m/2} d^m/dx^m P_n(x)
//
// Negative m via symmetry:
//   P_n^{-|m|}(x) = (-1)^{|m|} * (n-|m|)!/(n+|m|)! * P_n^{|m|}(x)
//
// Recurrence:
//   1. Seed: P_{|m|}^{|m|}(x) = (-1)^{|m|} * (2|m|-1)!! * (1-x^2)^{|m|/2}
//   2. Seed: P_{|m|+1}^{|m|}(x) = x * (2|m|+1) * P_{|m|}^{|m|}(x)
//   3. (k-|m|+1) P_{k+1}^{|m|}(x) = (2k+1) x P_k^{|m|}(x) - (k+|m|) P_{k-1}^{|m|}(x)

template <typename T>
C10_HOST_DEVICE T associated_legendre_polynomial_p(T n, T m, T x) {
  // Propagate NaN
  if (cmath_compat::isnan(n) || cmath_compat::isnan(m) || cmath_compat::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  int n_int = static_cast<int>(n);
  int m_int = static_cast<int>(m);
  int abs_m = std::abs(m_int);

  // Invalid: |m| > n or n < 0
  if (n_int < 0 || abs_m > n_int) {
    return T(0);
  }

  // Compute P_n^{|m|}(x) via recurrence
  // Seed: P_{|m|}^{|m|}(x) = (-1)^{|m|} * (2|m|-1)!! * (1-x^2)^{|m|/2}
  T pmm = T(1);
  if (abs_m > 0) {
    T somx2 = std::sqrt(T(1) - x * x);
    T fact = T(1);
    for (int i = 1; i <= abs_m; i++) {
      pmm *= -fact * somx2;  // Condon-Shortley phase
      fact += T(2);
    }
  }

  T result;
  if (n_int == abs_m) {
    result = pmm;
  } else {
    // Seed: P_{|m|+1}^{|m|}(x) = x * (2|m|+1) * P_{|m|}^{|m|}(x)
    T pmm1 = x * T(2 * abs_m + 1) * pmm;
    if (n_int == abs_m + 1) {
      result = pmm1;
    } else {
      // Recurrence: (k-|m|+1) P_{k+1}^{|m|} = (2k+1) x P_k^{|m|} - (k+|m|) P_{k-1}^{|m|}
      T p_prev = pmm;
      T p_curr = pmm1;
      for (int k = abs_m + 1; k < n_int; k++) {
        T p_next = (T(2 * k + 1) * x * p_curr - T(k + abs_m) * p_prev) / T(k - abs_m + 1);
        p_prev = p_curr;
        p_curr = p_next;
      }
      result = p_curr;
    }
  }

  // Handle negative m: P_n^{-|m|}(x) = (-1)^{|m|} * (n-|m|)!/(n+|m|)! * P_n^{|m|}(x)
  if (m_int < 0) {
    T sign = (abs_m % 2 == 0) ? T(1) : T(-1);
    T factor = T(1);
    for (int i = n_int - abs_m + 1; i <= n_int + abs_m; i++) {
      factor *= T(i);
    }
    result = sign * result / factor;
  }

  return result;
}

// Complex version: extract real parts for n, m; compute with real arithmetic
template <typename T>
C10_HOST_DEVICE c10::complex<T> associated_legendre_polynomial_p(c10::complex<T> n, c10::complex<T> m, c10::complex<T> x) {
  T result_real = associated_legendre_polynomial_p(n.real(), m.real(), x.real());
  return c10::complex<T>(result_real, T(0));
}

} // namespace torchscience::kernel::special_functions
