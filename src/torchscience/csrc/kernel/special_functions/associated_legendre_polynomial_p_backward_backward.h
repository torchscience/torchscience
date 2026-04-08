#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "associated_legendre_polynomial_p.h"
#include "associated_legendre_polynomial_p_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for associated Legendre polynomial P_n^m(x)
//
// Since grad_n = 0 and grad_m = 0 always, only gg_x contributes.
// Uses finite differences for second derivatives of grad_x w.r.t. n, m, x.

template <typename T>
std::tuple<T, T, T, T> associated_legendre_polynomial_p_backward_backward(
    T gg_n, T gg_m, T gg_x,
    T gradient, T n, T m, T x) {

  T gradient_gradient_output = T(0);
  T new_grad_n = T(0);
  T new_grad_m = T(0);
  T new_grad_x = T(0);

  // Since grad_n = 0 and grad_m = 0 always:
  // gg_n and gg_m contributions are zero (derivatives of zero).
  // Only gg_x contributes.

  if (std::abs(gg_x) > T(1e-15)) {
    T eps = T(1e-5);

    // d(grad_x)/d_gradient: the gradient formula with gradient=1
    auto [gn_u, gm_u, gx_u] = associated_legendre_polynomial_p_backward(T(1), n, m, x);
    gradient_gradient_output += gg_x * gx_u;

    // d(grad_x)/dn using finite difference
    auto [gn_np, gm_np, gx_np] = associated_legendre_polynomial_p_backward(gradient, n + eps, m, x);
    auto [gn_nm, gm_nm, gx_nm] = associated_legendre_polynomial_p_backward(gradient, n - eps, m, x);
    new_grad_n += gg_x * (gx_np - gx_nm) / (T(2) * eps);

    // d(grad_x)/dm using finite difference
    auto [gn_mp, gm_mp, gx_mp] = associated_legendre_polynomial_p_backward(gradient, n, m + eps, x);
    auto [gn_mm, gm_mm, gx_mm] = associated_legendre_polynomial_p_backward(gradient, n, m - eps, x);
    new_grad_m += gg_x * (gx_mp - gx_mm) / (T(2) * eps);

    // d(grad_x)/dx using finite difference (d^2P/dx^2)
    auto [gn_xp, gm_xp, gx_xp] = associated_legendre_polynomial_p_backward(gradient, n, m, x + eps);
    auto [gn_xm, gm_xm, gx_xm] = associated_legendre_polynomial_p_backward(gradient, n, m, x - eps);
    new_grad_x += gg_x * (gx_xp - gx_xm) / (T(2) * eps);
  }

  return {gradient_gradient_output, new_grad_n, new_grad_m, new_grad_x};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
associated_legendre_polynomial_p_backward_backward(
    c10::complex<T> gg_n, c10::complex<T> gg_m, c10::complex<T> gg_x,
    c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> m, c10::complex<T> x) {

  T eps_val = T(1e-5);
  c10::complex<T> eps(eps_val, T(0));
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  c10::complex<T> gradient_gradient_output = zero;
  c10::complex<T> new_grad_n = zero;
  c10::complex<T> new_grad_m = zero;
  c10::complex<T> new_grad_x = zero;

  if (std::abs(gg_x) > eps_val) {
    // d(grad_x)/d_gradient
    auto [gn_u, gm_u, gx_u] = associated_legendre_polynomial_p_backward(one, n, m, x);
    gradient_gradient_output += gg_x * std::conj(gx_u);

    // d(grad_x)/dn
    auto [gn_np, gm_np, gx_np] = associated_legendre_polynomial_p_backward(gradient, n + eps, m, x);
    auto [gn_nm, gm_nm, gx_nm] = associated_legendre_polynomial_p_backward(gradient, n - eps, m, x);
    new_grad_n += gg_x * std::conj((gx_np - gx_nm) / (two * eps));

    // d(grad_x)/dm
    auto [gn_mp, gm_mp, gx_mp] = associated_legendre_polynomial_p_backward(gradient, n, m + eps, x);
    auto [gn_mm, gm_mm, gx_mm] = associated_legendre_polynomial_p_backward(gradient, n, m - eps, x);
    new_grad_m += gg_x * std::conj((gx_mp - gx_mm) / (two * eps));

    // d(grad_x)/dx
    auto [gn_xp, gm_xp, gx_xp] = associated_legendre_polynomial_p_backward(gradient, n, m, x + eps);
    auto [gn_xm, gm_xm, gx_xm] = associated_legendre_polynomial_p_backward(gradient, n, m, x - eps);
    new_grad_x += gg_x * std::conj((gx_xp - gx_xm) / (two * eps));
  }

  return {gradient_gradient_output, new_grad_n, new_grad_m, new_grad_x};
}

} // namespace torchscience::kernel::special_functions
