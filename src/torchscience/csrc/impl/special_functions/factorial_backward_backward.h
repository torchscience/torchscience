#pragma once

/*
 * Factorial Function Double-Backward Pass (Second-Order Derivative)
 *
 * Computes second-order gradients for Hessian-vector products.
 *
 * d^2/dz^2 z! = Gamma(z + 1) * (psi(z + 1)^2 + psi'(z + 1))
 * where psi is the digamma function and psi' is the trigamma function.
 *
 * See factorial.h for the forward implementation.
 * See factorial_backward.h for the first-order derivative.
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <tuple>

#include "factorial_backward.h"
#include "trigamma.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Double-backward implementation (second-order derivative)
// ============================================================================

/**
 * Double-backward pass for factorial function.
 *
 * Given:
 *   gg_z = gradient w.r.t. gradient_z from first backward
 *   grad_output = original upstream gradient
 *   z = original input
 *
 * The first backward computes:
 *   gradient_z = grad_output * Gamma(z+1) * psi(z+1)
 *
 * Double backward computes:
 *   gradient_grad_output = gg_z * Gamma(z+1) * psi(z+1)
 *   gradient_z = gg_z * grad_output * d/dz[Gamma(z+1) * psi(z+1)]
 *              = gg_z * grad_output * Gamma(z+1) * (psi(z+1)^2 + psi'(z+1))
 *
 * Returns: (gradient_grad_output, gradient_z)
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<scalar_t, scalar_t>
factorial_backward_backward(
    scalar_t gg_z,
    scalar_t grad_output,
    scalar_t z,
    bool has_gg_z
) {
  scalar_t gradient_grad_output = scalar_t(0);
  scalar_t gradient_z = scalar_t(0);

  if (!has_gg_z) {
    return std::make_tuple(gradient_grad_output, gradient_z);
  }

  auto z_plus_1 = z + scalar_t(1);
  auto gamma_z_plus_1 = gamma(z_plus_1);
  auto psi_z_plus_1 = digamma(z_plus_1);
  auto psi_prime_z_plus_1 = trigamma(z_plus_1);

  // d/dz[Gamma(z+1) * psi(z+1)] = Gamma(z+1) * (psi(z+1)^2 + psi'(z+1))
  auto dGamma_psi_dz = gamma_z_plus_1 * (psi_z_plus_1 * psi_z_plus_1 + psi_prime_z_plus_1);

  if constexpr (c10::is_complex<scalar_t>::value) {
    // Wirtinger chain rule: conjugate each holomorphic derivative term
    gradient_grad_output = gg_z * std::conj(gamma_z_plus_1 * psi_z_plus_1);
    gradient_z = gg_z * grad_output * std::conj(dGamma_psi_dz);
  } else {
    gradient_grad_output = gg_z * gamma_z_plus_1 * psi_z_plus_1;
    gradient_z = gg_z * grad_output * dGamma_psi_dz;
  }

  return std::make_tuple(gradient_grad_output, gradient_z);
}

}  // namespace torchscience::impl::special_functions
