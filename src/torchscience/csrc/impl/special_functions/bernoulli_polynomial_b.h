#pragma once

#include <boost/math/special_functions/bernoulli.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Helper to compute all Bernoulli numbers B_k for k = 0, ..., n
// B_0 = 1, B_1 = -1/2, B_k = 0 for odd k > 1, B_{2m} from boost
template <typename T>
T bernoulli_number(int k) {
    if (k == 0) return T(1);
    if (k == 1) return T(-0.5);
    if (k % 2 == 1) return T(0);  // Odd Bernoulli numbers (k > 1) are zero
    return boost::math::bernoulli_b2n<T>(k / 2);
}

// Bernoulli polynomial B_n(x) = sum_{k=0}^{n} C(n,k) * B_k * x^(n-k)
template <typename T>
T bernoulli_polynomial_b(T n, T x) {
    int n_int = static_cast<int>(n);
    T result = T(0);
    for (int k = 0; k <= n_int; ++k) {
        T binom = boost::math::binomial_coefficient<T>(n_int, k);
        T bk = bernoulli_number<T>(k);
        T x_power = std::pow(x, n_int - k);
        result += binom * bk * x_power;
    }
    return result;
}

// Backward pass: d/dx B_n(x) = n * B_{n-1}(x)
template <typename T>
std::tuple<T, T> bernoulli_polynomial_b_backward(T n, T x) {
    // Gradient with respect to n is not well-defined (discrete parameter)
    T grad_n = T(0);

    // Derivative with respect to x: d/dx B_n(x) = n * B_{n-1}(x)
    int n_int = static_cast<int>(n);
    T grad_x;
    if (n_int == 0) {
        grad_x = T(0);  // B_0(x) = 1 is constant
    } else {
        grad_x = n * bernoulli_polynomial_b(T(n_int - 1), x);
    }

    return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
