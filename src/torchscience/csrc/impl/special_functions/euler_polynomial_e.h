#pragma once

#include <boost/math/special_functions/binomial.hpp>
#include <cmath>
#include <tuple>
#include <vector>

namespace torchscience::impl::special_functions {

// Helper to compute Euler numbers E_n
// E_0 = 1, E_1 = 0, E_2 = -1, E_3 = 0, E_4 = 5, ...
template <typename T>
T euler_number_for_poly(int n) {
    if (n % 2 != 0) {
        return T(0);
    }

    int m = n / 2;
    std::vector<T> E(m + 1);
    E[0] = T(1);

    for (int i = 1; i <= m; ++i) {
        T sum = T(0);
        T binom = T(1);
        for (int k = 0; k < i; ++k) {
            sum += binom * E[k];
            binom *= T((2*i - 2*k) * (2*i - 2*k - 1)) / T((2*k + 1) * (2*k + 2));
        }
        E[i] = -sum;
    }

    return E[m];
}

// Euler polynomial E_n(x)
// E_n(x) = sum_{k=0}^{n} C(n,k) * (E_k / 2^k) * (x - 1/2)^(n-k)
template <typename T>
T euler_polynomial_e(T n, T x) {
    int n_int = static_cast<int>(n);
    T result = T(0);
    T x_shifted = x - T(0.5);

    for (int k = 0; k <= n_int; ++k) {
        T binom = boost::math::binomial_coefficient<T>(n_int, k);
        T ek = euler_number_for_poly<T>(k);
        T two_pow_k = std::pow(T(2), k);
        T x_power = std::pow(x_shifted, n_int - k);
        result += binom * (ek / two_pow_k) * x_power;
    }
    return result;
}

// Backward pass: d/dx E_n(x) = n * E_{n-1}(x)
template <typename T>
std::tuple<T, T> euler_polynomial_e_backward(T n, T x) {
    // Gradient with respect to n is not well-defined (discrete parameter)
    T grad_n = T(0);

    // Derivative with respect to x: d/dx E_n(x) = n * E_{n-1}(x)
    int n_int = static_cast<int>(n);
    T grad_x;
    if (n_int == 0) {
        grad_x = T(0);  // E_0(x) = 1 is constant
    } else {
        grad_x = n * euler_polynomial_e(T(n_int - 1), x);
    }

    return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
