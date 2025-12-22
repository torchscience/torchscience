#pragma once

/*
 * Gauss Hypergeometric Function 2F1(a, b; c; z) - Second-Order Backward Pass
 *
 * This file contains the second-order backward pass for the hypergeometric function,
 * computing second derivatives (Hessian-vector products) for higher-order autodiff.
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <limits>

#include "hypergeometric_2_f_1.h"
#include "hypergeometric_2_f_1_backward.h"

namespace torchscience::impl::special_functions {

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
  using std::abs;

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

  // Check for unit circle divergence case
  const real_t unit_circle_tol = std::numeric_limits<real_t>::epsilon() * real_t(1000);
  real_t z_mag = abs(z);

  if (abs(z_mag - real_t(1)) < unit_circle_tol) {
    if (!check_unit_circle_convergence(a, b, c, z)) {
      // Series diverges: return NaN gradients
      real_t nan_val = std::numeric_limits<real_t>::quiet_NaN();
      return std::make_tuple(
        scalar_t(nan_val), scalar_t(nan_val), scalar_t(nan_val),
        scalar_t(nan_val), scalar_t(nan_val)
      );
    }
  }

  // Use adaptive step size for finite differences based on parameter magnitudes
  auto [h, h_unused] = compute_richardson_deltas_multi(a, b, c, z);
  (void)h_unused;  // Only need single step size for central differences

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
