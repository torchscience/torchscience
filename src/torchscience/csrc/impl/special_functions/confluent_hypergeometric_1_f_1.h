#pragma once

#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Confluent hypergeometric function 1F1(a, b, z), also known as M(a, b, z)
// or Kummer's confluent hypergeometric function
template <typename T>
T confluent_hypergeometric_1_f_1(T a, T b, T z) {
    return boost::math::hypergeometric_1F1(a, b, z);
}

template <typename T>
std::tuple<T, T, T> confluent_hypergeometric_1_f_1_backward(T a, T b, T z) {
    // Partial derivatives of 1F1(a, b, z)
    // d/dz 1F1(a, b, z) = (a/b) * 1F1(a+1, b+1, z)
    // For d/da and d/db, use numerical differentiation

    T h = T(1e-7);

    // Numerical gradient for a
    T f_a_plus = boost::math::hypergeometric_1F1(a + h, b, z);
    T f_a_minus = boost::math::hypergeometric_1F1(a - h, b, z);
    T grad_a = (f_a_plus - f_a_minus) / (T(2) * h);

    // Numerical gradient for b
    T f_b_plus = boost::math::hypergeometric_1F1(a, b + h, z);
    T f_b_minus = boost::math::hypergeometric_1F1(a, b - h, z);
    T grad_b = (f_b_plus - f_b_minus) / (T(2) * h);

    // Analytical gradient for z: d/dz 1F1(a, b, z) = (a/b) * 1F1(a+1, b+1, z)
    T grad_z;
    if (std::abs(b) < T(1e-10)) {
        // Avoid division by zero
        T f_z_plus = boost::math::hypergeometric_1F1(a, b, z + h);
        T f_z_minus = boost::math::hypergeometric_1F1(a, b, z - h);
        grad_z = (f_z_plus - f_z_minus) / (T(2) * h);
    } else {
        grad_z = (a / b) * boost::math::hypergeometric_1F1(a + T(1), b + T(1), z);
    }

    return std::make_tuple(grad_a, grad_b, grad_z);
}

} // namespace torchscience::impl::special_functions
