#pragma once

#include <boost/math/special_functions/hermite.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Probabilist's Hermite polynomial He_n(x)
// Related to physicist's Hermite polynomial H_n by:
// He_n(x) = 2^{-n/2} * H_n(x / sqrt(2))
template <typename T>
T hermite_polynomial_he(T n, T x) {
    unsigned int n_int = static_cast<unsigned int>(n);
    T scale = std::pow(T(2), -T(n_int) / T(2));
    T arg = x / std::sqrt(T(2));
    return scale * boost::math::hermite(n_int, arg);
}

template <typename T>
std::tuple<T, T> hermite_polynomial_he_backward(T n, T x) {
    // d/dx He_n(x) = n * He_{n-1}(x)
    T grad_n = T(0);
    unsigned int n_int = static_cast<unsigned int>(n);
    T grad_x;
    if (n_int == 0) {
        grad_x = T(0);
    } else {
        grad_x = n * hermite_polynomial_he(n - T(1), x);
    }
    return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
