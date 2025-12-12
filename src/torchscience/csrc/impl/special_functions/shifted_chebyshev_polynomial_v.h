#pragma once

#include <boost/math/special_functions/chebyshev.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Shifted Chebyshev polynomial of the third kind: V*_n(x) = V_n(2x - 1)
// Defined on [0, 1] instead of [-1, 1]
template <typename T>
T shifted_chebyshev_polynomial_v(T n, T x) {
    return boost::math::chebyshev_v(static_cast<unsigned>(n), T(2) * x - T(1));
}

template <typename T>
std::tuple<T, T> shifted_chebyshev_polynomial_v_backward(T n, T x) {
    // d/dx V*_n(x) = d/dx V_n(2x - 1) = 2 * V'_n(2x - 1)
    // Use numerical differentiation for simplicity
    T grad_n = T(0);
    unsigned int n_int = static_cast<unsigned int>(n);

    T h = T(1e-7);
    T f_plus = boost::math::chebyshev_v(n_int, T(2) * (x + h) - T(1));
    T f_minus = boost::math::chebyshev_v(n_int, T(2) * (x - h) - T(1));
    T grad_x = (f_plus - f_minus) / (T(2) * h);

    return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
