#pragma once

#include <boost/math/special_functions/legendre.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Associated Legendre polynomial P_n^m(x)
template <typename T>
T associated_legendre_p(T n, T m, T x) {
    return boost::math::legendre_p(static_cast<int>(n), static_cast<int>(m), x);
}

template <typename T>
std::tuple<T, T, T> associated_legendre_p_backward(T n, T m, T x) {
    // Gradients with respect to n and m are not well-defined (discrete parameters)
    T grad_n = T(0);
    T grad_m = T(0);

    // Use numerical differentiation for gradient w.r.t. x
    T h = T(1e-7);
    int n_int = static_cast<int>(n);
    int m_int = static_cast<int>(m);
    T f_plus = boost::math::legendre_p(n_int, m_int, x + h);
    T f_minus = boost::math::legendre_p(n_int, m_int, x - h);
    T grad_x = (f_plus - f_minus) / (T(2) * h);

    return std::make_tuple(grad_n, grad_m, grad_x);
}

} // namespace torchscience::impl::special_functions
