#pragma once

#include <cmath>

namespace torchscience::impl::special_functions {

// Euler's totient function phi(n)
// Counts the number of integers from 1 to n that are coprime to n
// phi(1) = 1
// phi(p) = p - 1 for prime p
// phi(p^k) = p^(k-1) * (p - 1) for prime power
// Uses the formula: phi(n) = n * product_{p | n} (1 - 1/p)
template <typename T>
T euler_totient_phi(T n_val) {
    int n = static_cast<int>(n_val);

    if (n <= 0) {
        return T(0);  // Not defined for non-positive integers
    }

    if (n == 1) {
        return T(1);
    }

    T result = T(n);
    int temp = n;

    // Find all prime factors and apply the formula
    // phi(n) = n * (1 - 1/p1) * (1 - 1/p2) * ... for each distinct prime factor
    for (int p = 2; p * p <= temp; ++p) {
        if (temp % p == 0) {
            // p is a prime factor
            // Apply (1 - 1/p) = (p - 1) / p
            result -= result / T(p);
            // Remove all factors of p
            while (temp % p == 0) {
                temp /= p;
            }
        }
    }

    // If temp > 1, then it's a prime factor
    if (temp > 1) {
        result -= result / T(temp);
    }

    return result;
}

template <typename T>
T euler_totient_phi_backward(T n) {
    // Euler's totient is defined only at integer points
    // Gradient is zero for discrete functions
    return T(0);
}

} // namespace torchscience::impl::special_functions
