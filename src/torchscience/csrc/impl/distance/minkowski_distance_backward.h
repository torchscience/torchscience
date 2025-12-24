#pragma once

/*
 * Minkowski Distance Backward Implementation
 *
 * GRADIENT DERIVATION:
 * ====================
 * Let d = ( sum_i w_i * |x_i - y_i|^p )^(1/p)
 *
 * Partial derivative with respect to x_k:
 *
 *   dd/dx_k = (1/p) * d^(1-p) * w_k * p * |x_k - y_k|^(p-1) * sign(x_k - y_k)
 *           = d^(1-p) * w_k * |x_k - y_k|^(p-1) * sign(x_k - y_k)
 *           = w_k * sign(x_k - y_k) * |x_k - y_k|^(p-1) / d^(p-1)
 *
 * Similarly: dd/dy_k = -dd/dx_k
 *
 * EDGE CASES:
 * ===========
 * - d = 0: gradient is 0 (all components identical)
 * - x_k = y_k: that component's gradient is 0
 * - p < 1: gradient can be large near zero (quasi-metric)
 */

#include <c10/macros/Macros.h>
#include <cmath>

namespace torchscience::impl::distance {

/**
 * Compute gradients for weighted Minkowski distance.
 *
 * @param grad_out Upstream gradient (scalar)
 * @param x First vector
 * @param y Second vector
 * @param d Dimension
 * @param p Order of norm
 * @param w Optional weights
 * @param dist Pre-computed distance value
 * @param grad_x Output gradient for x (d elements)
 * @param grad_y Output gradient for y (d elements)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void minkowski_distance_backward_pair(
    T grad_out,
    const T* x,
    const T* y,
    int64_t d,
    T p,
    const T* w,
    T dist,
    T* grad_x,
    T* grad_y
) {
    // Handle zero distance case
    if (dist == T(0)) {
        for (int64_t i = 0; i < d; ++i) {
            grad_x[i] = T(0);
            grad_y[i] = T(0);
        }
        return;
    }

    // Compute d^(p-1) for the denominator
    T dist_pow_pm1 = std::pow(dist, p - T(1));

    for (int64_t i = 0; i < d; ++i) {
        T diff = x[i] - y[i];

        // Handle zero difference
        if (diff == T(0)) {
            grad_x[i] = T(0);
            grad_y[i] = T(0);
            continue;
        }

        T abs_diff = std::abs(diff);
        T sign_diff = diff >= T(0) ? T(1) : T(-1);

        // |x_k - y_k|^(p-1)
        T abs_diff_pow_pm1;
        if (p == T(1)) {
            abs_diff_pow_pm1 = T(1);
        } else if (p == T(2)) {
            abs_diff_pow_pm1 = abs_diff;
        } else {
            abs_diff_pow_pm1 = std::pow(abs_diff, p - T(1));
        }

        // Weight factor
        T weight_i = (w != nullptr) ? w[i] : T(1);

        // Gradient: w_k * sign(diff) * |diff|^(p-1) / d^(p-1)
        T grad_component = weight_i * sign_diff * abs_diff_pow_pm1 / dist_pow_pm1;

        grad_x[i] = grad_out * grad_component;
        grad_y[i] = -grad_out * grad_component;
    }
}

}  // namespace torchscience::impl::distance
