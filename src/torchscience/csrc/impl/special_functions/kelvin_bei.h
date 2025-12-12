#pragma once

#include <boost/math/special_functions/kelvin.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Kelvin function bei_v(x) - imaginary part of J_v(x * exp(3*pi*i/4))
template <typename T>
T kelvin_bei(T v, T x) {
    return boost::math::kelvin_bei(v, x);
}

template <typename T>
std::tuple<T, T> kelvin_bei_backward(T v, T x) {
    // Gradient with respect to v is complex, use numerical differentiation
    T h = T(1e-7);
    T grad_v = (boost::math::kelvin_bei(v + h, x) - boost::math::kelvin_bei(v - h, x)) / (T(2) * h);

    // Derivative with respect to x using numerical differentiation
    T grad_x = (boost::math::kelvin_bei(v, x + h) - boost::math::kelvin_bei(v, x - h)) / (T(2) * h);

    return std::make_tuple(grad_v, grad_x);
}

} // namespace torchscience::impl::special_functions
