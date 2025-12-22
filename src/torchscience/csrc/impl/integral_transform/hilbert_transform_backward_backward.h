#pragma once

/*
 * Hilbert Transform Double Backward (Hessian-Vector Product) Implementation
 *
 * DERIVATION:
 * ===========
 * For double backward (computing Hessian-vector products), we need:
 *
 * Forward: y = H[x]
 * Backward: grad_x = -H[grad_y]
 *
 * Double backward computes gradients with respect to grad_y and x
 * given grad_grad_x (the vector being multiplied by the Hessian).
 *
 * Since H is a linear operator:
 *   - The Hessian of y with respect to x is zero (H is linear)
 *   - grad_grad_y = -H[grad_grad_x] (adjoint applied to incoming gradient)
 *   - new_grad_x = 0 (no second-order contribution from x)
 *
 * However, if there's a grad_y contribution:
 *   d/d(grad_y)[grad_x] = d/d(grad_y)[-H[grad_y]] = -H (constant w.r.t. grad_y)
 *
 * The double backward provides:
 *   grad_grad_output = H^T[grad_grad_input] = -H[grad_grad_input]
 *   new_grad_input = 0 (since H is linear, no second derivative contribution)
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

#include "hilbert_transform.h"
#include "hilbert_transform_backward.h"

namespace torchscience::impl::integral_transform {

/**
 * Compute double backward frequency response.
 *
 * For double backward, we apply H^T = -H to grad_grad_input to get grad_grad_output.
 * This is the same as the backward frequency response.
 *
 * @param k Frequency index (0 to n-1)
 * @param n Total number of frequency bins
 * @return Complex multiplier for double backward
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> hilbert_backward_backward_frequency_response(int64_t k, int64_t n) {
    // Same as backward: H^T = -H
    return hilbert_backward_frequency_response<T>(k, n);
}

/**
 * Apply double backward Hilbert frequency response in-place.
 *
 * @param spectrum Complex FFT output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_hilbert_backward_backward_response_inplace(c10::complex<T>* spectrum, int64_t n) {
    apply_hilbert_backward_response_inplace<T>(spectrum, n);
}

/**
 * Apply double backward Hilbert frequency response.
 *
 * @param input Complex FFT input array of length n
 * @param output Complex output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_hilbert_backward_backward_response(
    const c10::complex<T>* input,
    c10::complex<T>* output,
    int64_t n
) {
    apply_hilbert_backward_response<T>(input, output, n);
}

}  // namespace torchscience::impl::integral_transform
