#pragma once

#include <cmath>
#include <vector>

namespace torchscience::impl::special_functions {

// Euler numbers E_n
// E_0 = 1, E_1 = 0, E_2 = -1, E_3 = 0, E_4 = 5, E_5 = 0, E_6 = -61, ...
// Odd Euler numbers are all zero.
// Even Euler numbers alternate in sign.
template <typename T>
T euler_number_e(T n) {
    int n_int = static_cast<int>(n);

    // Odd Euler numbers are zero
    if (n_int % 2 != 0) {
        return T(0);
    }

    // For even n, compute E_n using the recurrence relation
    // E_0 = 1
    // For n >= 2 even: sum_{k=0}^{n} C(n,k) * E_k * 2^(n-k) = 0 where E_n appears with coefficient 1
    // Rearranging: E_n = -sum_{k=0}^{n-2, k even} C(n,k) * E_k * 2^(n-k)

    int m = n_int / 2;  // Index for even Euler numbers

    // Compute Euler numbers up to E_{2m}
    std::vector<T> E(m + 1);
    E[0] = T(1);  // E_0 = 1

    for (int i = 1; i <= m; ++i) {
        // Compute E_{2i} using the recurrence
        // Based on: sum_{k=0}^{i} C(2i, 2k) * E_{2k} = 0
        // So: E_{2i} = -sum_{k=0}^{i-1} C(2i, 2k) * E_{2k}
        T sum = T(0);
        T binom = T(1);  // C(2i, 0) = 1
        for (int k = 0; k < i; ++k) {
            sum += binom * E[k];
            // Update binomial coefficient: C(2i, 2k+2) = C(2i, 2k) * (2i-2k)(2i-2k-1) / ((2k+1)(2k+2))
            binom *= T((2*i - 2*k) * (2*i - 2*k - 1)) / T((2*k + 1) * (2*k + 2));
        }
        E[i] = -sum;
    }

    return E[m];
}

template <typename T>
T euler_number_e_backward(T n) {
    // Euler numbers are defined only at integer points
    // Gradient is zero for discrete functions
    return T(0);
}

} // namespace torchscience::impl::special_functions
