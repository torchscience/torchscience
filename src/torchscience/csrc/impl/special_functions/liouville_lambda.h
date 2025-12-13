#pragma once

#include <cmath>

namespace torchscience::impl::special_functions {

// Liouville function lambda(n)
// lambda(n) = (-1)^Omega(n) where Omega(n) is the number of prime factors
// of n counted with multiplicity
// lambda(1) = 1
// lambda(p) = -1 for any prime p
// lambda(p^k) = (-1)^k
// lambda is completely multiplicative: lambda(mn) = lambda(m)*lambda(n)
template <typename T>
T liouville_lambda(T n_val) {
    int n = static_cast<int>(n_val);

    if (n <= 0) {
        return T(0);  // Not defined for non-positive integers
    }

    if (n == 1) {
        return T(1);
    }

    int omega = 0;  // Total count of prime factors with multiplicity
    int temp = n;

    // Count prime factors with multiplicity
    for (int p = 2; p * p <= temp; ++p) {
        while (temp % p == 0) {
            temp /= p;
            omega++;
        }
    }

    // If temp > 1, then it's a prime factor
    if (temp > 1) {
        omega++;
    }

    // Return (-1)^omega
    return (omega % 2 == 0) ? T(1) : T(-1);
}

template <typename T>
T liouville_lambda_backward(T n) {
    // Liouville function is defined only at integer points
    // Gradient is zero for discrete functions
    return T(0);
}

} // namespace torchscience::impl::special_functions
