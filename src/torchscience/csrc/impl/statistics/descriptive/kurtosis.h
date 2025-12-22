#pragma once

/*
 * Kurtosis Implementation
 *
 * MATHEMATICAL DEFINITION:
 * ========================
 * Kurtosis measures the "tailedness" of a probability distribution.
 *
 * For a sample x_1, x_2, ..., x_n:
 *
 * Biased (sample) kurtosis:
 *   g_2 = m_4 / m_2^2 - 3  (excess, fisher=true)
 *   g_2 = m_4 / m_2^2      (Pearson, fisher=false)
 *
 * where:
 *   m_2 = (1/n) * sum((x_i - mean)^2)  (biased variance)
 *   m_4 = (1/n) * sum((x_i - mean)^4)  (biased 4th central moment)
 *
 * Unbiased (population) kurtosis (bias=false):
 *   G_2 = ((n-1) / ((n-2)(n-3))) * ((n+1)*g_2 + 6)
 *
 * ALGORITHM:
 * ==========
 * We use a numerically stable two-pass algorithm:
 *   Pass 1: Compute mean
 *   Pass 2: Compute m_2 and m_4 using (x - mean)
 *
 * For complex tensors, we compute kurtosis of magnitudes |z|.
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::impl::descriptive {

// ============================================================================
// Forward: Compute kurtosis for a contiguous array
// ============================================================================

/**
 * Compute kurtosis of a 1D array using two-pass algorithm.
 *
 * @param data Input array of n elements
 * @param n Number of elements
 * @param fisher If true, compute excess kurtosis (subtract 3)
 * @param bias If true, return biased estimate; if false, apply correction
 * @return Kurtosis value (or NaN if n <= 3 for unbiased, or n < 2)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T kurtosis_1d(
    const T* data,
    int64_t n,
    bool fisher,
    bool bias
) {
    // Handle edge cases
    if (n < 2) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (!bias && n <= 3) {
        // Unbiased kurtosis requires n > 3
        return std::numeric_limits<T>::quiet_NaN();
    }

    // Pass 1: Compute mean
    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

    // Pass 2: Compute central moments m_2 and m_4
    T m2 = T(0);
    T m4 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        T d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= T(n);
    m4 /= T(n);

    // Handle zero variance case
    if (m2 == T(0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // Compute biased kurtosis
    T m2_sq = m2 * m2;
    T g2 = m4 / m2_sq;
    if (fisher) {
        g2 -= T(3);
    }

    // Apply bias correction if requested
    if (!bias) {
        // G_2 = ((n-1) / ((n-2)(n-3))) * ((n+1)*g_2 + 6)
        T n_f = T(n);
        T correction = ((n_f - T(1)) / ((n_f - T(2)) * (n_f - T(3)))) *
                      ((n_f + T(1)) * g2 + T(6));
        return correction;
    }

    return g2;
}

/**
 * Compute kurtosis of complex magnitudes.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T kurtosis_1d_complex(
    const c10::complex<T>* data,
    int64_t n,
    bool fisher,
    bool bias
) {
    if (n < 2) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (!bias && n <= 3) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // Pass 1: Compute mean of magnitudes
    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += std::abs(data[i]);
    }
    T mean = sum / T(n);

    // Pass 2: Compute central moments of magnitudes
    T m2 = T(0);
    T m4 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T mag = std::abs(data[i]);
        T d = mag - mean;
        T d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= T(n);
    m4 /= T(n);

    if (m2 == T(0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T m2_sq = m2 * m2;
    T g2 = m4 / m2_sq;
    if (fisher) {
        g2 -= T(3);
    }

    if (!bias) {
        T n_f = T(n);
        T correction = ((n_f - T(1)) / ((n_f - T(2)) * (n_f - T(3)))) *
                      ((n_f + T(1)) * g2 + T(6));
        return correction;
    }

    return g2;
}

}  // namespace torchscience::impl::descriptive
