#pragma once

#include <boost/math/special_functions/kelvin.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Kelvin function kei_v(x) - imaginary part of K_v(x * exp(pi*i/4))
template <typename T>
T kelvin_kei(T v, T x) {
    return boost::math::kelvin_kei(v, x);
}

template <typename T>
std::tuple<T, T> kelvin_kei_backward(T v, T x) {
    // Gradient with respect to v is complex, use numerical differentiation
    T h = T(1e-7);
    T grad_v = (boost::math::kelvin_kei(v + h, x) - boost::math::kelvin_kei(v - h, x)) / (T(2) * h);

    // Derivative with respect to x using numerical differentiation
    T grad_x = (boost::math::kelvin_kei(v, x + h) - boost::math::kelvin_kei(v, x - h)) / (T(2) * h);

    return std::make_tuple(grad_v, grad_x);
}

} // namespace torchscience::impl::special_functions
