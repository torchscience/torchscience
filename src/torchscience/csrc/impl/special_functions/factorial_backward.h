#pragma once

/*
 * Factorial Function Backward Pass (First-Order Derivative)
 *
 * Computes: d/dz z! = Gamma(z + 1) * psi(z + 1)
 * where psi is the digamma function.
 *
 * See factorial.h for the forward implementation.
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

#include "factorial.h"
#include "gamma.h"
#include "digamma.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Backward implementation (first-order derivative)
// ============================================================================

/**
 * Backward pass for factorial function.
 *
 * Since z! = Gamma(z + 1):
 *   d/dz z! = Gamma(z + 1) * psi(z + 1)
 *
 * where psi is the digamma function.
 *
 * Returns gradient with respect to z.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t
factorial_backward(scalar_t grad_output, scalar_t z) {
  auto z_plus_1 = z + scalar_t(1);
  auto gamma_z_plus_1 = gamma(z_plus_1);
  auto psi_z_plus_1 = digamma(z_plus_1);

  if constexpr (c10::is_complex<scalar_t>::value) {
    // Wirtinger chain rule: conjugate the holomorphic derivative
    return grad_output * std::conj(gamma_z_plus_1 * psi_z_plus_1);
  } else {
    return grad_output * gamma_z_plus_1 * psi_z_plus_1;
  }
}

}  // namespace torchscience::impl::special_functions
