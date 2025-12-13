#pragma once

#include <cmath>

namespace torchscience::impl::special_functions {

// Mobius function mu(n)
// mu(1) = 1
// mu(n) = (-1)^k if n is a product of k distinct primes (square-free)
// mu(n) = 0 if n has a squared prime factor
template <typename T>
T mobius_mu(T n_val) {
    int n = static_cast<int>(n_val);

    if (n <= 0) {
        return T(0);  // Not defined for non-positive integers
    }

    if (n == 1) {
        return T(1);
    }

    int prime_factors = 0;
    int temp = n;

    // Check for each prime factor
    for (int p = 2; p * p <= temp; ++p) {
        if (temp % p == 0) {
            temp /= p;
            prime_factors++;

            // If p divides temp again, n is not square-free
            if (temp % p == 0) {
                return T(0);
            }
        }
    }

    // If temp > 1, then it's a prime factor
    if (temp > 1) {
        prime_factors++;
    }

    // Return (-1)^k
    return (prime_factors % 2 == 0) ? T(1) : T(-1);
}

template <typename T>
T mobius_mu_backward(T n) {
    // Mobius function is defined only at integer points
    // Gradient is zero for discrete functions
    return T(0);
}

} // namespace torchscience::impl::special_functions
