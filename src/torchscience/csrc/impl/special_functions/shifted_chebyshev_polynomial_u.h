#pragma once

#include <boost/math/special_functions/chebyshev.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Shifted Chebyshev polynomial of the second kind: U*_n(x) = U_n(2x - 1)
// Defined on [0, 1] instead of [-1, 1]
template <typename T>
T shifted_chebyshev_polynomial_u(T n, T x) {
    return boost::math::chebyshev_u(static_cast<unsigned>(n), T(2) * x - T(1));
}

template <typename T>
std::tuple<T, T> shifted_chebyshev_polynomial_u_backward(T n, T x) {
    // d/dx U*_n(x) = d/dx U_n(2x - 1) = 2 * U'_n(2x - 1)
    // U'_n(y) = ((n+1)*T_{n+1}(y) - y*U_n(y)) / (y^2 - 1) for y != ±1
    // For shifted version: y = 2x - 1
    T grad_n = T(0);
    unsigned int n_int = static_cast<unsigned int>(n);
    T y = T(2) * x - T(1);
    T grad_x;

    T y_sq_minus_1 = y * y - T(1);
    if (std::abs(y_sq_minus_1) < T(1e-10)) {
        // At boundaries, use numerical differentiation
        T h = T(1e-7);
        T f_plus = boost::math::chebyshev_u(n_int, T(2) * (x + h) - T(1));
        T f_minus = boost::math::chebyshev_u(n_int, T(2) * (x - h) - T(1));
        grad_x = (f_plus - f_minus) / (T(2) * h);
    } else {
        T T_np1 = boost::math::chebyshev_t(n_int + 1, y);
        T U_n = boost::math::chebyshev_u(n_int, y);
        T U_prime_y = ((n + T(1)) * T_np1 - y * U_n) / y_sq_minus_1;
        grad_x = T(2) * U_prime_y;
    }

    return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
