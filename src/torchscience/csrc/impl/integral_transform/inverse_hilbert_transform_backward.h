#pragma once

/*
 * Inverse Hilbert Transform Backward Implementation
 *
 * GRADIENT DERIVATION:
 * ====================
 * The inverse Hilbert transform is H^{-1}[f] = -H[f].
 *
 * Forward: y = H^{-1}[x] = -H[x]
 * Backward: grad_x = (H^{-1})^T[grad_y] = -H^T[grad_y] = -(-H)[grad_y] = H[grad_y]
 *
 * Since (H^{-1})^T = (-H)^T = -H^T = -(-H) = H
 *
 * Therefore: grad_input = H[grad_output]
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

#include "inverse_hilbert_transform.h"
#include "hilbert_transform.h"

namespace torchscience::impl::integral_transform {

/**
 * Compute the backward frequency response multiplier for inverse Hilbert transform.
 *
 * For backward pass of H^{-1}, we need (H^{-1})^T = H.
 * So the frequency response is the same as forward Hilbert: h[k] = -i * sign(freq[k])
 *
 * @param k Frequency index (0 to n-1)
 * @param n Total number of frequency bins
 * @return Complex multiplier for backward pass
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> inverse_hilbert_backward_frequency_response(int64_t k, int64_t n) {
    // Backward of H^{-1} is H, so use forward Hilbert response
    return hilbert_frequency_response<T>(k, n);
}

/**
 * Apply backward inverse Hilbert frequency response in-place.
 *
 * This applies H (the adjoint of H^{-1}) in the frequency domain.
 *
 * @param spectrum Complex FFT output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_inverse_hilbert_backward_response_inplace(c10::complex<T>* spectrum, int64_t n) {
    // Use forward Hilbert response since (H^{-1})^T = H
    apply_hilbert_response_inplace<T>(spectrum, n);
}

/**
 * Apply backward inverse Hilbert frequency response to spectrum.
 *
 * @param input Complex FFT input array of length n
 * @param output Complex output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_inverse_hilbert_backward_response(
    const c10::complex<T>* input,
    c10::complex<T>* output,
    int64_t n
) {
    // Use forward Hilbert response since (H^{-1})^T = H
    apply_hilbert_response<T>(input, output, n);
}

}  // namespace torchscience::impl::integral_transform
