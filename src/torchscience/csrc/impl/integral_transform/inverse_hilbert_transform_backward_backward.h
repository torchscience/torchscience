#pragma once

/*
 * Inverse Hilbert Transform Double Backward Implementation
 *
 * DERIVATION:
 * ===========
 * For double backward (computing Hessian-vector products):
 *
 * Forward: y = H^{-1}[x] = -H[x]
 * Backward: grad_x = H[grad_y]
 *
 * Double backward computes gradients with respect to grad_y and x
 * given grad_grad_x (the vector being multiplied by the Hessian).
 *
 * Since H^{-1} is a linear operator:
 *   - grad_grad_y = H^T[grad_grad_x] = -H[grad_grad_x]
 *   - new_grad_x = 0 (no second-order contribution)
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

#include "inverse_hilbert_transform.h"
#include "inverse_hilbert_transform_backward.h"
#include "hilbert_transform_backward.h"

namespace torchscience::impl::integral_transform {

/**
 * Compute double backward frequency response for inverse Hilbert transform.
 *
 * For double backward, we apply H^T = -H to grad_grad_input to get grad_grad_output.
 *
 * @param k Frequency index (0 to n-1)
 * @param n Total number of frequency bins
 * @return Complex multiplier for double backward
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> inverse_hilbert_backward_backward_frequency_response(int64_t k, int64_t n) {
    // H^T = -H, same as hilbert_backward_frequency_response
    return hilbert_backward_frequency_response<T>(k, n);
}

/**
 * Apply double backward inverse Hilbert frequency response in-place.
 *
 * @param spectrum Complex FFT output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_inverse_hilbert_backward_backward_response_inplace(c10::complex<T>* spectrum, int64_t n) {
    apply_hilbert_backward_response_inplace<T>(spectrum, n);
}

/**
 * Apply double backward inverse Hilbert frequency response.
 *
 * @param input Complex FFT input array of length n
 * @param output Complex output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_inverse_hilbert_backward_backward_response(
    const c10::complex<T>* input,
    c10::complex<T>* output,
    int64_t n
) {
    apply_hilbert_backward_response<T>(input, output, n);
}

}  // namespace torchscience::impl::integral_transform
