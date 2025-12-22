#pragma once

/*
 * Gamma Function Double-Backward Pass (Second-Order Derivative)
 *
 * Computes second-order gradients for Hessian-vector products.
 *
 * d^2/dz^2 Gamma(z) = Gamma(z) * (psi(z)^2 + psi'(z))
 * where psi(z) is the digamma function and psi'(z) is the trigamma function.
 *
 * See gamma.h for the forward implementation.
 * See gamma_backward.h for the first-order derivative.
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <tuple>

#include "gamma_backward.h"
#include "trigamma.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Double-backward implementation (second-order derivative)
// ============================================================================

/**
 * Double-backward pass for gamma function.
 *
 * Given:
 *   gradient_gradient_z = gradient w.r.t. gradient_z from first backward
 *   gradient_output = original upstream gradient
 *   z = original input
 *
 * The first backward computes:
 *   gradient_z = gradient_output * Gamma(z) * psi(z)
 *
 * Double backward computes:
 *   gradient_gradient_output = gradient_gradient_z * Gamma(z) * psi(z)  (derivative w.r.t gradient_output)
 *   gradient_z = gradient_gradient_z * gradient_output * d/dz[Gamma(z) * psi(z)]
 *              = gradient_gradient_z * gradient_output * Gamma(z) * (psi(z)^2 + psi'(z))
 *
 * Returns: (gradient_gradient_output, gradient_z)
 *
 * Complex Gradient Convention (Wirtinger Calculus):
 * ------------------------------------------------
 * Same convention as gamma_backward applies here. For each term that involves
 * a holomorphic derivative, we conjugate when computing the Wirtinger gradient.
 *
 * gradient_gradient_output: This is dL/d(gradient_output)_bar, and since the first backward
 *   computed gradient_z = gradient_output * [Gamma(z)*psi(z)], differentiating w.r.t.
 *   gradient_output (treating it as a variable) gives conj(Gamma(z)*psi(z)).
 *
 * gradient_z: This differentiates through the [Gamma(z)*psi(z)] term, giving
 *   conj(d/dz[Gamma(z)*psi(z)]) = conj(Gamma(z)*(psi(z)^2 + psi'(z))).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T> gamma_backward_backward(
  T gradient_gradient_z,
  T gradient_output,
  T z,
  const bool has_gradient_gradient_z
) {
  T gradient_gradient_output;
  T gradient_z;

  if (!has_gradient_gradient_z) {
    return std::make_tuple(T(0), T(0));
  }

  if constexpr (c10::is_complex<T>::value) {
    gradient_gradient_output = gradient_gradient_z * std::conj(gamma(z) * digamma(z));

    gradient_z = gradient_gradient_z * gradient_output * std::conj(gamma(z) * (digamma(z) * digamma(z) + trigamma(z)));
  } else {
    gradient_gradient_output = gradient_gradient_z * gamma(z) * digamma(z);

    gradient_z = gradient_gradient_z * gradient_output * (gamma(z) * (digamma(z) * digamma(z) + trigamma(z)));
  }

  return std::make_tuple(
    gradient_gradient_output,
    gradient_z
  );
}

}  // namespace torchscience::impl::special_functions
