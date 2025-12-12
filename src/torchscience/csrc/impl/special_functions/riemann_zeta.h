#pragma once

#include <boost/math/special_functions/zeta.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

// Riemann zeta function zeta(s)
// zeta(s) = sum_{n=1}^{inf} 1/n^s for Re(s) > 1
template <typename T>
T riemann_zeta(T s) {
    return boost::math::zeta(s);
}

template <typename T>
T riemann_zeta_backward(T s) {
    // Derivative of zeta(s) is -sum_{n=2}^{inf} ln(n)/n^s
    // Use numerical differentiation
    T h = T(1e-7);
    T f_plus = boost::math::zeta(s + h);
    T f_minus = boost::math::zeta(s - h);
    return (f_plus - f_minus) / (T(2) * h);
}

} // namespace torchscience::impl::special_functions
