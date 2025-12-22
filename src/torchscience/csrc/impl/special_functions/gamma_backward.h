#pragma once

/*
 * Gamma Function Backward Pass (First-Order Derivative)
 *
 * Computes: d/dz Gamma(z) = Gamma(z) * psi(z)
 * where psi(z) is the digamma function.
 *
 * See gamma.h for the forward implementation.
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

#include "gamma.h"
#include "digamma.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Backward implementation (first-order derivative)
// ============================================================================

/**
 * Backward pass for gamma function.
 * d/dz Gamma(z) = Gamma(z) * psi(z)
 *
 * Returns gradient with respect to z.
 *
 * Complex Gradient Convention (Wirtinger Calculus):
 * ------------------------------------------------
 * PyTorch stores dL/dz_bar (conjugate Wirtinger derivative) in .grad, not dL/dz.
 * This convention makes gradient descent work correctly for complex parameters,
 * since the steepest descent direction for a real loss L is 2*dL/dz_bar.
 *
 * For a holomorphic function f(z), the chain rule in Wirtinger calculus is:
 *     dL/dz_bar = (dL/dw_bar) * conj(df/dz)
 * where w = f(z).
 *
 * For Gamma(z), the holomorphic derivative is dGamma/dz = Gamma(z)*psi(z), so:
 *     gradient_z_bar = gradient_w_bar * conj(Gamma(z)*psi(z))
 *
 * For real types, conjugation is the identity, so no special handling needed.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T gamma_backward(
  T gradient_output,
  T z
) {
  if constexpr (c10::is_complex<T>::value) {
    return gradient_output * std::conj(gamma(z) * digamma(z));
  }

  return gradient_output * gamma(z) * digamma(z);
}

}  // namespace torchscience::impl::special_functions
