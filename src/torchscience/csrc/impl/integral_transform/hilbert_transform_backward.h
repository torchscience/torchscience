#pragma once

/*
 * Hilbert Transform Backward Implementation
 *
 * GRADIENT DERIVATION:
 * ====================
 * The Hilbert transform is a linear operator, so its gradient is straightforward.
 *
 * Forward: y = H[x]
 * Backward: grad_x = H[grad_y]
 *
 * This is because H is orthogonal (preserves inner products):
 *   <H[f], g> = <f, H^T[g]> = <f, -H[g]>
 *
 * But for the gradient of a loss L with respect to input x:
 *   dL/dx = H^T[dL/dy] = -H[dL/dy]
 *
 * Wait, let's be more careful. For real-valued functions:
 *   H^T = -H (the adjoint of H is -H)
 *
 * So: grad_input = -H[grad_output]
 *
 * However, in practice with FFT implementation:
 *   H[f] = IFFT(FFT(f) * h)
 *   H^T[f] = IFFT(FFT(f) * conj(h)) = IFFT(FFT(f) * (-h)) = -H[f]
 *
 * Therefore: grad_input = -H[grad_output] = H^{-1}[grad_output]
 *
 * Since H[H[f]] = -f, we have H^{-1} = -H, confirming:
 *   grad_input = -H[grad_output]
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

#include "hilbert_transform.h"

namespace torchscience::impl::integral_transform {

/**
 * Compute the backward frequency response multiplier.
 *
 * For backward pass, we need the adjoint of H, which is -H.
 * So the frequency response is negated: h_backward[k] = -h[k] = i * sign(freq[k])
 *
 * @param k Frequency index (0 to n-1)
 * @param n Total number of frequency bins
 * @return Complex multiplier for backward pass
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> hilbert_backward_frequency_response(int64_t k, int64_t n) {
    // DC component: still 0
    if (k == 0) {
        return c10::complex<T>(T(0), T(0));
    }

    // Nyquist frequency for even-length signals: still 0
    if (n % 2 == 0 && k == n / 2) {
        return c10::complex<T>(T(0), T(0));
    }

    // Positive frequencies: h_backward[k] = +i (negated from forward -i)
    if (k < (n + 1) / 2) {
        return c10::complex<T>(T(0), T(1));
    }

    // Negative frequencies: h_backward[k] = -i (negated from forward +i)
    return c10::complex<T>(T(0), T(-1));
}

/**
 * Apply backward Hilbert frequency response in-place.
 *
 * This applies -H (the adjoint of H) in the frequency domain.
 *
 * @param spectrum Complex FFT output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_hilbert_backward_response_inplace(c10::complex<T>* spectrum, int64_t n) {
    for (int64_t k = 0; k < n; ++k) {
        c10::complex<T> h = hilbert_backward_frequency_response<T>(k, n);
        spectrum[k] = spectrum[k] * h;
    }
}

/**
 * Apply backward Hilbert frequency response to spectrum.
 *
 * @param input Complex FFT input array of length n
 * @param output Complex output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_hilbert_backward_response(
    const c10::complex<T>* input,
    c10::complex<T>* output,
    int64_t n
) {
    for (int64_t k = 0; k < n; ++k) {
        c10::complex<T> h = hilbert_backward_frequency_response<T>(k, n);
        output[k] = input[k] * h;
    }
}

}  // namespace torchscience::impl::integral_transform
