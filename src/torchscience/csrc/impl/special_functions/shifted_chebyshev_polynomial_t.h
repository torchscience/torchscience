#pragma once

#include <boost/math/special_functions/chebyshev.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Shifted Chebyshev polynomial of the first kind: T*_n(x) = T_n(2x - 1)
// Defined on [0, 1] instead of [-1, 1]
template <typename T>
T shifted_chebyshev_polynomial_t(T n, T x) {
    return boost::math::chebyshev_t(static_cast<unsigned>(n), T(2) * x - T(1));
}

template <typename T>
std::tuple<T, T> shifted_chebyshev_polynomial_t_backward(T n, T x) {
    // d/dx T*_n(x) = d/dx T_n(2x - 1) = 2 * T'_n(2x - 1)
    // T'_n(y) = n * U_{n-1}(y), so:
    // d/dx T*_n(x) = 2 * n * U_{n-1}(2x - 1)
    T grad_n = T(0);
    unsigned int n_int = static_cast<unsigned int>(n);
    T grad_x;
    if (n_int == 0) {
        grad_x = T(0);
    } else {
        T y = T(2) * x - T(1);
        grad_x = T(2) * n * boost::math::chebyshev_u(n_int - 1, y);
    }
    return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
