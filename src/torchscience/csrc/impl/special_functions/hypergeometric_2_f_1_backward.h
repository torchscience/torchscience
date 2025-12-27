#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "hypergeometric_2_f_1.h"

namespace torchscience::impl::special_functions {

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

  const real_t unit_circle_tol = std::numeric_limits<real_t>::epsilon() * real_t(1000);

  real_t z_mag = abs(z);
  scalar_t one_minus_z = scalar_t(1) - z;
  real_t one_minus_z_mag = abs(one_minus_z);

  // Check for unit circle divergence case
  if (abs(z_mag - real_t(1)) < unit_circle_tol) {
    if (!check_unit_circle_convergence(a, b, c, z)) {
      // Series diverges: return NaN gradients
      real_t nan_val = std::numeric_limits<real_t>::quiet_NaN();
      return std::make_tuple(
        scalar_t(nan_val), scalar_t(nan_val),
        scalar_t(nan_val), scalar_t(nan_val)
      );
    }

    // Special case: z = 1 (within tolerance)
    // Use Gauss summation theorem derivatives
    if (one_minus_z_mag < unit_circle_tol) {
      auto [value, da, db, dc] = hypergeometric_2f1_gauss_summation_with_derivatives(a, b, c);
      (void)value;

      // For z derivative at z=1: d/dz 2F1(a,b;c;z) = (ab/c) * 2F1(a+1,b+1;c+1;z)
      // At z=1, this uses Gauss summation if Re(c-a-b-1) > 0
      scalar_t cab = c - a - b;
      real_t re_cab;
      if constexpr (c10::is_complex<scalar_t>::value) {
        re_cab = cab.real();
      } else {
        re_cab = cab;
      }

      scalar_t d_z;
      if (re_cab > real_t(1)) {
        // Gauss summation applies to the derivative at z=1
        scalar_t coeff = (a * b) / c;
        d_z = coeff * hypergeometric_2f1_gauss_summation(a + scalar_t(1), b + scalar_t(1), c + scalar_t(1));
      } else {
        // Derivative series at z=1 doesn't converge; use limit from below
        // Approximate using z very close to 1
        scalar_t z_approx = scalar_t(real_t(1) - unit_circle_tol * real_t(10));
        d_z = hypergeometric_2f1_derivative(a, b, c, z_approx);
      }

      scalar_t gradient_a, gradient_b, gradient_c, gradient_z;
      if constexpr (c10::is_complex<scalar_t>::value) {
        gradient_a = gradient_output * std::conj(da);
        gradient_b = gradient_output * std::conj(db);
        gradient_c = gradient_output * std::conj(dc);
        gradient_z = gradient_output * std::conj(d_z);
      } else {
        gradient_a = gradient_output * da;
        gradient_b = gradient_output * db;
        gradient_c = gradient_output * dc;
        gradient_z = gradient_output * d_z;
      }

      return std::make_tuple(gradient_a, gradient_b, gradient_c, gradient_z);
    }
  }

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
    auto [
      value,
      da,
      db,
      dc
    ] = hypergeometric_2f1_series_with_param_derivatives(
      a,
      b,
      c,
      z
    );

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

}  // namespace torchscience::impl::special_functions
