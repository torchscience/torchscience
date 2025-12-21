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

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

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

// ============================================================================
// Double Backward: Second-order derivatives
// ============================================================================

/**
 * Compute second-order derivatives for kurtosis.
 *
 * Given:
 *   - grad_grad_input: d^2L/dx_i dx_j (incoming gradient from second backward)
 *   - grad_output: dL/dk (from first backward)
 *   - data: original input
 *
 * Returns:
 *   - grad_grad_output: d^2L/dk^2 contribution
 *   - new_grad_input: updated gradient for input
 *
 * The Hessian of kurtosis is complex, involving third and fourth moments.
 * For simplicity, we implement this using numerical stability considerations.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void kurtosis_backward_backward_1d(
    const T* grad_grad_input,
    T grad_output,
    const T* data,
    int64_t n,
    bool fisher,
    bool bias,
    T& grad_grad_output,
    T* new_grad_input
) {
    // Initialize outputs
    grad_grad_output = T(0);
    for (int64_t i = 0; i < n; ++i) {
        new_grad_input[i] = T(0);
    }

    if (n < 2 || (!bias && n <= 3)) {
        return;
    }

    // Compute forward quantities
    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

    T m2 = T(0);
    T m3 = T(0);
    T m4 = T(0);
    T m5 = T(0);
    T m6 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        T d2 = d * d;
        T d3 = d2 * d;
        T d4 = d2 * d2;
        m2 += d2;
        m3 += d3;
        m4 += d4;
        m5 += d4 * d;
        m6 += d3 * d3;
    }
    m2 /= T(n);
    m3 /= T(n);
    m4 /= T(n);
    m5 /= T(n);
    m6 /= T(n);

    if (m2 == T(0)) {
        return;
    }

    T m2_sq = m2 * m2;
    T m2_cb = m2_sq * m2;
    T g2 = m4 / m2_sq;
    T g2_excess = g2 - T(3);
    T k_for_grad = fisher ? g2_excess : g2;

    T n_f = T(n);
    T coeff = T(4) / (n_f * m2_sq);

    T dG2_dg2 = T(1);
    if (!bias) {
        dG2_dg2 = ((n_f - T(1)) * (n_f + T(1))) / ((n_f - T(2)) * (n_f - T(3)));
    }

    // Compute d^2k/dx_i dx_j contributions
    // This is complex - we compute the main diagonal and off-diagonal contributions

    // First, compute the gradient of gradient coefficients
    // dk/dx_i = coeff * (d_i^3 - m_3 - 2*k*m_2*d_i)
    // d^2k/dx_i^2 involves derivatives of d_i, m_2, m_3, m_4, and k

    // For simplicity and numerical stability, we compute:
    // d(dk/dx_i)/dx_j for the Hessian-vector product

    // The contribution to grad_grad_output from grad_grad_input:
    // grad_grad_output += sum_i grad_grad_input[i] * dk/dx_i
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        T d3 = d * d * d;
        T dk_dxi = coeff * (d3 - m3 - T(2) * k_for_grad * m2 * d);
        if (!bias) {
            dk_dxi *= dG2_dg2;
        }
        grad_grad_output += grad_grad_input[i] * dk_dxi;
    }

    // The contribution to new_grad_input from Hessian:
    // For each j: new_grad_input[j] = grad_output * sum_i grad_grad_input[i] * d^2k/dx_i dx_j

    // d^2k/dx_i dx_j involves:
    // - diagonal (i=j): d(dk/dx_i)/dx_i
    // - off-diagonal: d(dk/dx_i)/dx_j

    // Due to the dependence through mean, m2, m3, m4, this is complex.
    // We compute this term by term.

    // Sum of grad_grad_input for mean derivatives
    T sum_gg = T(0);
    T sum_gg_d = T(0);
    T sum_gg_d2 = T(0);
    T sum_gg_d3 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        sum_gg += grad_grad_input[i];
        sum_gg_d += grad_grad_input[i] * d;
        sum_gg_d2 += grad_grad_input[i] * d * d;
        sum_gg_d3 += grad_grad_input[i] * d * d * d;
    }

    // Compute Hessian-vector product
    for (int64_t j = 0; j < n; ++j) {
        T dj = data[j] - mean;
        T dj2 = dj * dj;
        T dj3 = dj2 * dj;

        // Contribution from d(d_i^3)/dx_j = 3*d_i^2 * (delta_ij - 1/n)
        T term1 = T(0);
        for (int64_t i = 0; i < n; ++i) {
            T di = data[i] - mean;
            T di2 = di * di;
            T kronecker = (i == j) ? T(1) : T(0);
            term1 += grad_grad_input[i] * T(3) * di2 * (kronecker - T(1) / n_f);
        }

        // Contribution from d(m_3)/dx_j = (3/n) * (d_j^2 - m_2)
        // minus sign because of -m_3 in the formula
        T dm3_dxj = (T(3) / n_f) * (dj2 - m2);
        T term2 = -sum_gg * dm3_dxj;

        // Contribution from d(2*k*m_2*d_i)/dx_j
        // = 2 * [dk/dx_j * m_2 * d_i + k * dm_2/dx_j * d_i + k * m_2 * d(d_i)/dx_j]
        // dm_2/dx_j = (2/n) * (d_j - 0) = (2/n) * d_j (since mean correction is 0 for centered)
        // Actually dm_2/dx_j = (2/n) * d_j for the centered moment

        T dk_dxj = coeff * (dj3 - m3 - T(2) * k_for_grad * m2 * dj);
        T dm2_dxj = (T(2) / n_f) * dj;

        T term3a = T(0);  // From dk/dx_j * m_2 * d_i
        T term3b = T(0);  // From k * dm_2/dx_j * d_i
        T term3c = T(0);  // From k * m_2 * d(d_i)/dx_j

        for (int64_t i = 0; i < n; ++i) {
            T di = data[i] - mean;
            T kronecker = (i == j) ? T(1) : T(0);

            term3a += grad_grad_input[i] * dk_dxj * m2 * di;
            term3b += grad_grad_input[i] * k_for_grad * dm2_dxj * di;
            term3c += grad_grad_input[i] * k_for_grad * m2 * (kronecker - T(1) / n_f);
        }
        T term3 = -T(2) * (term3a + term3b + term3c);

        // Contribution from d(coeff)/dx_j
        // coeff = 4 / (n * m_2^2)
        // d(coeff)/dx_j = -8 / (n * m_2^3) * dm_2/dx_j = -2 * coeff / m_2 * dm_2/dx_j
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

}  // namespace torchscience::impl::descriptive
