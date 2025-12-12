#pragma once

#include <boost/math/special_functions/legendre.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Legendre function of the second kind Q_n(x)
template <typename T>
T legendre_q(T n, T x) {
    return boost::math::legendre_q(static_cast<unsigned>(n), x);
}

template <typename T>
std::tuple<T, T> legendre_q_backward(T n, T x) {
    // Gradient with respect to n is not well-defined (discrete parameter)
    T grad_n = T(0);

    // Use numerical differentiation for gradient w.r.t. x
    T h = T(1e-7);
    unsigned n_int = static_cast<unsigned>(n);
    T f_plus = boost::math::legendre_q(n_int, x + h);
    T f_minus = boost::math::legendre_q(n_int, x - h);
    T grad_x = (f_plus - f_minus) / (T(2) * h);

    return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
