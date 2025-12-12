#pragma once

#include <boost/math/special_functions/legendre.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Legendre polynomial of the first kind P_n(x)
template <typename T>
T legendre_p(T n, T x) {
    return boost::math::legendre_p(static_cast<int>(n), x);
}

template <typename T>
std::tuple<T, T> legendre_p_backward(T n, T x) {
    // Gradient with respect to n is not well-defined (discrete parameter)
    T grad_n = T(0);

    // Derivative: dP_n/dx = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)
    // For |x| = 1, use the formula: dP_n/dx(1) = n*(n+1)/2, dP_n/dx(-1) = (-1)^{n+1} * n*(n+1)/2
    int n_int = static_cast<int>(n);
    T grad_x;

    if (n_int == 0) {
        grad_x = T(0);
    } else if (std::abs(x - T(1)) < T(1e-10)) {
        grad_x = n * (n + T(1)) / T(2);
    } else if (std::abs(x + T(1)) < T(1e-10)) {
        grad_x = (n_int % 2 == 0 ? T(-1) : T(1)) * n * (n + T(1)) / T(2);
    } else {
        T P_n = boost::math::legendre_p(n_int, x);
        T P_n_minus_1 = boost::math::legendre_p(n_int - 1, x);
        grad_x = n * (x * P_n - P_n_minus_1) / (x * x - T(1));
    }

    return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
