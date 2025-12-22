#pragma once

/*
 * Hilbert Transform Implementation
 *
 * MATHEMATICAL DEFINITION:
 * ========================
 * The Hilbert transform of a function f(t) is defined as:
 *
 *   H[f](x) = (1/pi) * PV integral from -inf to inf of f(t)/(t-x) dt
 *
 * where PV denotes the Cauchy principal value.
 *
 * DISCRETE IMPLEMENTATION:
 * ========================
 * For discrete signals, the Hilbert transform is computed via FFT:
 *
 *   1. Compute FFT: F[k] = FFT(f)
 *   2. Multiply by frequency response: H[k] = F[k] * h[k]
 *      where h[k] = -i * sign(freq[k])
 *        - h[0] = 0 (DC component)
 *        - h[k] = -i for positive frequencies (k = 1, ..., N/2-1)
 *        - h[N/2] = 0 for even N (Nyquist)
 *        - h[k] = +i for negative frequencies (k = N/2+1, ..., N-1)
 *   3. Compute IFFT: H[f] = IFFT(H)
 *
 * KEY PROPERTIES:
 * ===============
 * 1. H[sin(wt)] = cos(wt), H[cos(wt)] = -sin(wt)
 * 2. H[H[f]] = -f (involutory up to sign)
 * 3. Energy preservation: integral |H[f]|^2 = integral |f|^2
 * 4. Linearity: H[af + bg] = aH[f] + bH[g]
 * 5. Inverse: H^{-1}[f] = -H[f]
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::impl::integral_transform {

/**
 * Compute the Hilbert transform frequency response multiplier for index k.
 *
 * @param k Frequency index (0 to n-1)
 * @param n Total number of frequency bins
 * @return Complex multiplier h[k] = -i * sign(freq[k])
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> hilbert_frequency_response(int64_t k, int64_t n) {
    // DC component: h[0] = 0
    if (k == 0) {
        return c10::complex<T>(T(0), T(0));
    }

    // Nyquist frequency for even-length signals: h[n/2] = 0
    if (n % 2 == 0 && k == n / 2) {
        return c10::complex<T>(T(0), T(0));
    }

    // Positive frequencies: h[k] = -i (standard Hilbert transform convention)
    // k = 1, 2, ..., floor((n-1)/2)
    // h = -i * sign(freq) = -i * 1 = -i for positive frequencies
    if (k < (n + 1) / 2) {
        return c10::complex<T>(T(0), T(-1));
    }

    // Negative frequencies: h[k] = +i
    // h = -i * sign(freq) = -i * (-1) = +i for negative frequencies
    // k = ceil((n+1)/2), ..., n-1
    return c10::complex<T>(T(0), T(1));
}

/**
 * Compute sign factor for Hilbert transform.
 *
 * Returns:
 *  0 for DC and Nyquist
 * -1 for positive frequencies (h = -i means imaginary part is -1)
 * +1 for negative frequencies (h = +i means imaginary part is +1)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T hilbert_sign_factor(int64_t k, int64_t n) {
    if (k == 0) {
        return T(0);
    }
    if (n % 2 == 0 && k == n / 2) {
        return T(0);
    }
    if (k < (n + 1) / 2) {
        return T(-1);
    }
    return T(1);
}

/**
 * Apply Hilbert frequency response in-place to a complex spectrum.
 *
 * @param spectrum Complex FFT output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_hilbert_response_inplace(c10::complex<T>* spectrum, int64_t n) {
    for (int64_t k = 0; k < n; ++k) {
        c10::complex<T> h = hilbert_frequency_response<T>(k, n);
        spectrum[k] = spectrum[k] * h;
    }
}

/**
 * Apply Hilbert frequency response, storing result in separate output.
 *
 * @param input Complex FFT input array of length n
 * @param output Complex output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_hilbert_response(
    const c10::complex<T>* input,
    c10::complex<T>* output,
    int64_t n
) {
    for (int64_t k = 0; k < n; ++k) {
        c10::complex<T> h = hilbert_frequency_response<T>(k, n);
        output[k] = input[k] * h;
    }
}

/**
 * Compute the analytic signal from real input.
 *
 * The analytic signal is z(t) = f(t) + i*H[f](t)
 * where H[f] is the Hilbert transform.
 *
 * In frequency domain: Z[k] = F[k] * (1 + sign(freq[k]))
 *   - Z[0] = F[0] (DC preserved)
 *   - Z[k] = 2*F[k] for positive frequencies
 *   - Z[k] = 0 for negative frequencies
 *   - Z[n/2] = F[n/2] for even n (Nyquist preserved)
 *
 * @param k Frequency index
 * @param n Total number of frequency bins
 * @return Scale factor for analytic signal computation
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T analytic_signal_scale(int64_t k, int64_t n) {
    if (k == 0) {
        return T(1);  // DC
    }
    if (n % 2 == 0 && k == n / 2) {
        return T(1);  // Nyquist
    }
    if (k < (n + 1) / 2) {
        return T(2);  // Positive frequencies
    }
    return T(0);  // Negative frequencies
}

}  // namespace torchscience::impl::integral_transform
