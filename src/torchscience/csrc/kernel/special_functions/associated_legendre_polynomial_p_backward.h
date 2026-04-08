#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "associated_legendre_polynomial_p.h"

namespace torchscience::kernel::special_functions {

// Backward pass for associated Legendre polynomial P_n^m(x)
//
// Gradients:
//   grad_n = 0 (discrete parameter)
//   grad_m = 0 (discrete parameter)
//   grad_x = grad_output * [n*x*P_n^m(x) - (n+m)*P_{n-1}^m(x)] / (x^2 - 1)
//
// Singularity at x = +/-1 handled by clamping x away from endpoints.

template <typename T>
std::tuple<T, T, T> associated_legendre_polynomial_p_backward(T gradient, T n, T m, T x) {
  int n_int = static_cast<int>(n);
  int m_int = static_cast<int>(m);
  int abs_m = std::abs(m_int);

  T grad_x;

  if (n_int < 0 || abs_m > n_int) {
    grad_x = T(0);
  } else {
    T eps = std::numeric_limits<T>::epsilon();

    // Use integer-valued n and m to match the forward kernel's behavior
    T n_eff = T(n_int);
    T m_eff = T(m_int);

    // Clamp x away from +/-1 to avoid division by zero in x^2 - 1
    T x_c = x;
    T denom = x * x - T(1);
    if (std::abs(denom) < eps * T(100)) {
      x_c = (x >= T(0)) ? (T(1) - eps * T(100)) : (T(-1) + eps * T(100));
      denom = x_c * x_c - T(1);
    }

    T P_n_m = associated_legendre_polynomial_p(n_eff, m_eff, x_c);

    // P_{n-1}^m: when n-1 < |m|, the forward kernel returns 0
    T P_nm1_m = associated_legendre_polynomial_p(n_eff - T(1), m_eff, x_c);

    grad_x = (n_eff * x_c * P_n_m - (n_eff + m_eff) * P_nm1_m) / denom;
  }

  return {T(0), T(0), gradient * grad_x};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
associated_legendre_polynomial_p_backward(c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> m, c10::complex<T> x) {
  // Compute gradient using real arithmetic
  auto [gn_r, gm_r, gx_r] = associated_legendre_polynomial_p_backward(T(1), n.real(), m.real(), x.real());

  c10::complex<T> zero(T(0), T(0));
  return {zero, zero, gradient * std::conj(c10::complex<T>(gx_r, T(0)))};
}

} // namespace torchscience::kernel::special_functions
