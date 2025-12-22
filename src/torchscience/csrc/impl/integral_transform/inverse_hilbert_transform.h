#pragma once

/*
 * Inverse Hilbert Transform Implementation
 *
 * MATHEMATICAL DEFINITION:
 * ========================
 * The inverse Hilbert transform is defined as:
 *
 *   H^{-1}[f](x) = -H[f](x)
 *
 * This follows from the property that H[H[f]] = -f, which means H^{-1} = -H.
 *
 * DISCRETE IMPLEMENTATION:
 * ========================
 * For discrete signals, the inverse Hilbert transform is computed via FFT:
 *
 *   1. Compute FFT: F[k] = FFT(f)
 *   2. Multiply by inverse frequency response: H^{-1}[k] = F[k] * h_inv[k]
 *      where h_inv[k] = i * sign(freq[k]) = -h[k]
 *        - h_inv[0] = 0 (DC component)
 *        - h_inv[k] = +i for positive frequencies (k = 1, ..., N/2-1)
 *        - h_inv[N/2] = 0 for even N (Nyquist)
 *        - h_inv[k] = -i for negative frequencies (k = N/2+1, ..., N-1)
 *   3. Compute IFFT: H^{-1}[f] = IFFT(H^{-1})
 *
 * KEY PROPERTIES:
 * ===============
 * 1. H^{-1}[H[f]] = f
 * 2. H[H^{-1}[f]] = f
 * 3. H^{-1}[f] = -H[f]
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::impl::integral_transform {

/**
 * Compute the inverse Hilbert transform frequency response multiplier for index k.
 *
 * @param k Frequency index (0 to n-1)
 * @param n Total number of frequency bins
 * @return Complex multiplier h_inv[k] = i * sign(freq[k]) = -h[k]
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> inverse_hilbert_frequency_response(int64_t k, int64_t n) {
    // DC component: h_inv[0] = 0
    if (k == 0) {
        return c10::complex<T>(T(0), T(0));
    }

    // Nyquist frequency for even-length signals: h_inv[n/2] = 0
    if (n % 2 == 0 && k == n / 2) {
        return c10::complex<T>(T(0), T(0));
    }

    // Positive frequencies: h_inv[k] = +i (opposite of forward -i)
    if (k < (n + 1) / 2) {
        return c10::complex<T>(T(0), T(1));
    }

    // Negative frequencies: h_inv[k] = -i (opposite of forward +i)
    return c10::complex<T>(T(0), T(-1));
}

/**
 * Apply inverse Hilbert frequency response in-place to a complex spectrum.
 *
 * @param spectrum Complex FFT output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_inverse_hilbert_response_inplace(c10::complex<T>* spectrum, int64_t n) {
    for (int64_t k = 0; k < n; ++k) {
        c10::complex<T> h = inverse_hilbert_frequency_response<T>(k, n);
        spectrum[k] = spectrum[k] * h;
    }
}

/**
 * Apply inverse Hilbert frequency response, storing result in separate output.
 *
 * @param input Complex FFT input array of length n
 * @param output Complex output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_inverse_hilbert_response(
    const c10::complex<T>* input,
    c10::complex<T>* output,
    int64_t n
) {
    for (int64_t k = 0; k < n; ++k) {
        c10::complex<T> h = inverse_hilbert_frequency_response<T>(k, n);
        output[k] = input[k] * h;
    }
}

}  // namespace torchscience::impl::integral_transform
