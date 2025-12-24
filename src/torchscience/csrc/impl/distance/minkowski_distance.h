#pragma once

/*
 * Minkowski Distance Implementation
 *
 * MATHEMATICAL DEFINITION:
 * ========================
 * The weighted Minkowski distance between vectors x and y:
 *
 *   d_p(x, y; w) = ( sum_i w_i * |x_i - y_i|^p )^(1/p)
 *
 * Special cases:
 *   p = 1: Manhattan distance (weighted L1)
 *   p = 2: Euclidean distance (weighted L2)
 *   p -> inf: Chebyshev distance (max weighted absolute diff)
 *
 * ALGORITHM:
 * ==========
 * For numerical stability with large/small p:
 *   - For moderate p: direct computation
 *   - For very large p: approximate with max (TODO: future optimization)
 */

#include <c10/macros/Macros.h>
#include <cmath>

namespace torchscience::impl::distance {

/**
 * Compute weighted Minkowski distance between two vectors.
 *
 * @param x First vector (pointer to d elements)
 * @param y Second vector (pointer to d elements)
 * @param d Dimension of vectors
 * @param p Order of the norm (p > 0)
 * @param w Optional weights (nullptr for unweighted, else pointer to d elements)
 * @return Distance value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T minkowski_distance_pair(
    const T* x,
    const T* y,
    int64_t d,
    T p,
    const T* w
) {
    T sum = T(0);

    if (w == nullptr) {
        // Unweighted case
        for (int64_t i = 0; i < d; ++i) {
            T diff = x[i] - y[i];
            T abs_diff = diff >= T(0) ? diff : -diff;
            sum += std::pow(abs_diff, p);
        }
    } else {
        // Weighted case
        for (int64_t i = 0; i < d; ++i) {
            T diff = x[i] - y[i];
            T abs_diff = diff >= T(0) ? diff : -diff;
            sum += w[i] * std::pow(abs_diff, p);
        }
    }

    // Handle p = 1 and p = 2 specially for numerical stability
    if (p == T(1)) {
        return sum;
    } else if (p == T(2)) {
        return std::sqrt(sum);
    } else {
        return std::pow(sum, T(1) / p);
    }
}

}  // namespace torchscience::impl::distance
