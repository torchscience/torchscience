#pragma once

#include <boost/math/special_functions/polygamma.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Polygamma function psi^(n)(x) - the nth derivative of digamma
template <typename T>
T polygamma(T n, T x) {
    return boost::math::polygamma(static_cast<int>(n), x);
}

template <typename T>
std::tuple<T, T> polygamma_backward(T n, T x) {
    // Gradient with respect to n is not well-defined (discrete parameter)
    T grad_n = T(0);

    // Derivative with respect to x: d/dx psi^(n)(x) = psi^(n+1)(x)
    int n_int = static_cast<int>(n);
    T grad_x = boost::math::polygamma(n_int + 1, x);

    return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
