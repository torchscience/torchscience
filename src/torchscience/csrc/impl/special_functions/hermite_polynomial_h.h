#pragma once

#include <boost/math/special_functions/hermite.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Physicist's Hermite polynomial H_n(x)
// Satisfies the recurrence: H_{n+1}(x) = 2x*H_n(x) - 2n*H_{n-1}(x)
// with H_0(x) = 1 and H_1(x) = 2x
template <typename T>
T hermite_polynomial_h(T n, T x) {
    return boost::math::hermite(static_cast<unsigned int>(n), x);
}

template <typename T>
std::tuple<T, T> hermite_polynomial_h_backward(T n, T x) {
    // d/dx H_n(x) = 2n * H_{n-1}(x)
    T grad_n = T(0);
    unsigned int n_int = static_cast<unsigned int>(n);
    T grad_x;
    if (n_int == 0) {
        grad_x = T(0);
    } else {
        grad_x = T(2) * n * boost::math::hermite(n_int - 1, x);
    }
    return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
