// src/torchscience/csrc/kernel/statistics/descriptive/kurtosis.h
#pragma once

#include <cmath>
#include <limits>

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

namespace torchscience::kernel::statistics::descriptive {

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
T kurtosis(
    const T* data,
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

    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

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

/**
 * Compute kurtosis of complex magnitudes.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T kurtosis_complex(
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

    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += std::abs(data[i]);
    }
    T mean = sum / T(n);

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

// ============================================================================
// Backward: Compute gradient of kurtosis w.r.t. input
// ============================================================================

/**
 * Compute gradient of kurtosis w.r.t. input array.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void kurtosis_backward(
    T grad_output,
    const T* data,
    int64_t n,
    bool fisher,
    bool bias,
    T* grad_input
) {
    if (n < 2 || (!bias && n <= 3)) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = T(0);
        }
        return;
    }

    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

    T m2 = T(0);
    T m3 = T(0);
    T m4 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        T d2 = d * d;
        T d3 = d2 * d;
        m2 += d2;
        m3 += d3;
        m4 += d2 * d2;
    }
    m2 /= T(n);
    m3 /= T(n);
    m4 /= T(n);

    if (m2 == T(0)) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = T(0);
        }
        return;
    }

    T m2_sq = m2 * m2;
    T g2 = m4 / m2_sq;
    T coeff = T(4) / (T(n) * m2_sq);

    if (bias) {
        for (int64_t i = 0; i < n; ++i) {
            T d = data[i] - mean;
            T d3 = d * d * d;
            grad_input[i] = grad_output * coeff * (d3 - m3 - g2 * m2 * d);
        }
    } else {
        T n_f = T(n);
        T dG2_dg2 = ((n_f - T(1)) * (n_f + T(1))) / ((n_f - T(2)) * (n_f - T(3)));

        for (int64_t i = 0; i < n; ++i) {
            T d = data[i] - mean;
            T d3 = d * d * d;
            T dg2_dxi = coeff * (d3 - m3 - g2 * m2 * d);
            grad_input[i] = grad_output * dG2_dg2 * dg2_dxi;
        }
    }
}

/**
 * Compute gradient for complex input (magnitudes).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void kurtosis_backward_complex(
    T grad_output,
    const c10::complex<T>* data,
    int64_t n,
    bool fisher,
    bool bias,
    c10::complex<T>* grad_input
) {
    if (n < 2 || (!bias && n <= 3)) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = c10::complex<T>(T(0), T(0));
        }
        return;
    }

    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += std::abs(data[i]);
    }
    T mean = sum / T(n);

    T m2 = T(0);
    T m3 = T(0);
    T m4 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T mag = std::abs(data[i]);
        T d = mag - mean;
        T d2 = d * d;
        T d3 = d2 * d;
        m2 += d2;
        m3 += d3;
        m4 += d2 * d2;
    }
    m2 /= T(n);
    m3 /= T(n);
    m4 /= T(n);

    if (m2 == T(0)) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = c10::complex<T>(T(0), T(0));
        }
        return;
    }

    T m2_sq = m2 * m2;
    T g2 = m4 / m2_sq;
    T g2_excess = g2 - T(3);

    T k_for_grad = fisher ? g2_excess : g2;
    T coeff = T(4) / (T(n) * m2_sq);

    T dG2_dg2 = T(1);
    if (!bias) {
        T n_f = T(n);
        dG2_dg2 = ((n_f - T(1)) * (n_f + T(1))) / ((n_f - T(2)) * (n_f - T(3)));
    }

    for (int64_t i = 0; i < n; ++i) {
        T mag = std::abs(data[i]);
        T d = mag - mean;
        T d3 = d * d * d;
        T dk_dmag = coeff * (d3 - m3 - T(2) * k_for_grad * m2 * d);
        if (!bias) {
            dk_dmag *= dG2_dg2;
        }

        if (mag > T(0)) {
            T grad_mag = grad_output * dk_dmag;
            grad_input[i] = c10::complex<T>(
                grad_mag * data[i].real() / mag,
                grad_mag * data[i].imag() / mag
            );
        } else {
            grad_input[i] = c10::complex<T>(T(0), T(0));
        }
    }
}

// ============================================================================
// Double Backward: Second-order derivatives
// ============================================================================

/**
 * Compute second-order derivatives for kurtosis.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void kurtosis_backward_backward(
    const T* grad_grad_input,
    T grad_output,
    const T* data,
    int64_t n,
    bool fisher,
    bool bias,
    T& grad_grad_output,
    T* new_grad_input
) {
    grad_grad_output = T(0);
    for (int64_t i = 0; i < n; ++i) {
        new_grad_input[i] = T(0);
    }

    if (n < 2 || (!bias && n <= 3)) {
        return;
    }

    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

    T m2 = T(0);
    T m3 = T(0);
    T m4 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        T d2 = d * d;
        T d3 = d2 * d;
        T d4 = d2 * d2;
        m2 += d2;
        m3 += d3;
        m4 += d4;
    }
    m2 /= T(n);
    m3 /= T(n);
    m4 /= T(n);

    if (m2 == T(0)) {
        return;
    }

    T m2_sq = m2 * m2;
    T g2 = m4 / m2_sq;
    T g2_excess = g2 - T(3);
    T k_for_grad = fisher ? g2_excess : g2;

    T n_f = T(n);
    T coeff = T(4) / (n_f * m2_sq);

    T dG2_dg2 = T(1);
    if (!bias) {
        dG2_dg2 = ((n_f - T(1)) * (n_f + T(1))) / ((n_f - T(2)) * (n_f - T(3)));
    }

    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        T d3 = d * d * d;
        T dk_dxi = coeff * (d3 - m3 - T(2) * k_for_grad * m2 * d);
        if (!bias) {
            dk_dxi *= dG2_dg2;
        }
        grad_grad_output += grad_grad_input[i] * dk_dxi;
    }

    T sum_gg = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum_gg += grad_grad_input[i];
    }

    for (int64_t j = 0; j < n; ++j) {
        T dj = data[j] - mean;
        T dj2 = dj * dj;
        T dj3 = dj2 * dj;

        T term1 = T(0);
        for (int64_t i = 0; i < n; ++i) {
            T di = data[i] - mean;
            T di2 = di * di;
            T kronecker = (i == j) ? T(1) : T(0);
            term1 += grad_grad_input[i] * T(3) * di2 * (kronecker - T(1) / n_f);
        }

        T dm3_dxj = (T(3) / n_f) * (dj2 - m2);
        T term2 = -sum_gg * dm3_dxj;

        T dk_dxj = coeff * (dj3 - m3 - T(2) * k_for_grad * m2 * dj);
        T dm2_dxj = (T(2) / n_f) * dj;

        T term3a = T(0);
        T term3b = T(0);
        T term3c = T(0);

        for (int64_t i = 0; i < n; ++i) {
            T di = data[i] - mean;
            T kronecker = (i == j) ? T(1) : T(0);

            term3a += grad_grad_input[i] * dk_dxj * m2 * di;
            term3b += grad_grad_input[i] * k_for_grad * dm2_dxj * di;
            term3c += grad_grad_input[i] * k_for_grad * m2 * (kronecker - T(1) / n_f);
        }
        T term3 = -T(2) * (term3a + term3b + term3c);

        T dcoeff_dxj = -T(2) * coeff / m2 * dm2_dxj;
        T term4 = T(0);
        for (int64_t i = 0; i < n; ++i) {
            T di = data[i] - mean;
            T di3 = di * di * di;
            term4 += grad_grad_input[i] * (di3 - m3 - T(2) * k_for_grad * m2 * di);
        }
        term4 *= dcoeff_dxj;

        T hessian_contribution = coeff * (term1 + term2 + term3) + term4;

        if (!bias) {
            hessian_contribution *= dG2_dg2;
        }

        new_grad_input[j] = grad_output * hessian_contribution;
    }
}

}  // namespace torchscience::kernel::statistics::descriptive
