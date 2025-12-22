#pragma once

/*
 * Kurtosis Backward (First-order derivatives)
 *
 * GRADIENT COMPUTATION:
 * ====================
 * For biased excess kurtosis: k = m_4 / m_2^2 - 3
 *
 * Let d_i = x_i - mean
 * m_2 = (1/n) * sum(d_i^2)
 * m_4 = (1/n) * sum(d_i^4)
 *
 * dk/dx_i = (1/n) * [4*d_i^3/m_2^2 - 4*d_i*m_4/m_2^3]
 *         = (4/n) * d_i * [d_i^2/m_2^2 - m_4/m_2^3]
 *         = (4/n) * d_i * [d_i^2 - k*m_2] / m_2^2
 *
 * With mean gradient correction (since mean depends on all x_i):
 * dk/dx_i = (4/n) * [(d_i^3 - mean(d^3))/m_2^2 - 2*k*(d_i - 0)/m_2]
 *         = (4/n) * [(d_i^3 - m_3)/m_2^2 - 2*k*d_i/m_2]
 *
 * where m_3 = (1/n) * sum(d_i^3) is the 3rd central moment.
 *
 * Simplifying:
 * dk/dx_i = (4/n) * [d_i^3/m_2^2 - m_3/m_2^2 - 2*k*d_i/m_2]
 *         = (4/(n*m_2^2)) * [d_i^3 - m_3 - 2*k*m_2*d_i]
 *
 * For Pearson kurtosis (fisher=false), the formula is similar but without -3.
 * For unbiased kurtosis (bias=false), apply chain rule through G_2 formula.
 */

#include "kurtosis.h"
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::impl::descriptive {

// ============================================================================
// Backward: Compute gradient of kurtosis w.r.t. input
// ============================================================================

/**
 * Compute gradient of kurtosis w.r.t. input array.
 *
 * The gradient for biased excess kurtosis k = m_4/m_2^2 - 3 is:
 *   dk/dx_i = (4/(n*m_2^2)) * [d_i^3 - m_3 - 2*(k+3)*m_2*d_i]
 *
 * For Pearson kurtosis (fisher=false), use k instead of (k+3).
 *
 * @param grad_output Upstream gradient (scalar)
 * @param data Input array of n elements
 * @param n Number of elements
 * @param fisher If true, excess kurtosis was computed
 * @param bias If true, biased estimate was used
 * @param grad_input Output gradient array of n elements
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void kurtosis_backward_1d(
    T grad_output,
    const T* data,
    int64_t n,
    bool fisher,
    bool bias,
    T* grad_input
) {
    // Handle edge cases - no gradient
    if (n < 2 || (!bias && n <= 3)) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = T(0);
        }
        return;
    }

    // Recompute forward quantities
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

    // Handle zero variance
    if (m2 == T(0)) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = T(0);
        }
        return;
    }

    T m2_sq = m2 * m2;
    T g2 = m4 / m2_sq;  // Pearson kurtosis (m4/m2^2)

    // Gradient derivation:
    // k = m_4/m_2^2 - 3 (for fisher=true) or k = m_4/m_2^2 (for fisher=false)
    //
    // Using chain rule:
    // dk/dx_j = d(m_4)/dx_j / m_2^2 - 2*m_4/m_2^3 * d(m_2)/dx_j
    //
    // where d(m_2)/dx_j = (2/n)*d_j and d(m_4)/dx_j = (4/n)*(d_j^3 - m_3)
    //
    // dk/dx_j = (4/n)*(d_j^3 - m_3)/m_2^2 - 2*m_4/m_2^3 * (2/n)*d_j
    //         = (4/(n*m_2^2)) * [(d_j^3 - m_3) - m_4*d_j/m_2]
    //         = (4/(n*m_2^2)) * [d_j^3 - m_3 - g2*m_2*d_j]
    //
    // Note: The gradient is the same for fisher and non-fisher (the -3 is constant)

    T coeff = T(4) / (T(n) * m2_sq);

    if (bias) {
        // Biased case: direct gradient
        for (int64_t i = 0; i < n; ++i) {
            T d = data[i] - mean;
            T d3 = d * d * d;
            grad_input[i] = grad_output * coeff * (d3 - m3 - g2 * m2 * d);
        }
    } else {
        // Unbiased case: chain rule through correction formula
        // G_2 = ((n-1) / ((n-2)(n-3))) * ((n+1)*g_2 + 6)
        // dG_2/dg_2 = ((n-1)(n+1)) / ((n-2)(n-3))
        T n_f = T(n);
        T dG2_dg2 = ((n_f - T(1)) * (n_f + T(1))) / ((n_f - T(2)) * (n_f - T(3)));

        for (int64_t i = 0; i < n; ++i) {
            T d = data[i] - mean;
            T d3 = d * d * d;
            // Gradient of biased kurtosis
            T dg2_dxi = coeff * (d3 - m3 - g2 * m2 * d);
            // Chain rule
            grad_input[i] = grad_output * dG2_dg2 * dg2_dxi;
        }
    }
}

/**
 * Compute gradient for complex input (magnitudes).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void kurtosis_backward_1d_complex(
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

    // Compute magnitudes and their mean
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

        // Gradient of |z| w.r.t. z: d|z|/dz = conj(z) / (2|z|)
        // For real gradient: d|z|/d(Re(z)) = Re(z)/|z|, d|z|/d(Im(z)) = Im(z)/|z|
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

}  // namespace torchscience::impl::descriptive
