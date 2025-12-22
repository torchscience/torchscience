#pragma once

/*
 * Kurtosis Double Backward (Second-order derivatives)
 *
 * SECOND-ORDER DERIVATIVES:
 * =========================
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

#include "kurtosis.h"
#include "kurtosis_backward.h"
#include <c10/macros/Macros.h>
#include <cmath>

namespace torchscience::impl::descriptive {

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
