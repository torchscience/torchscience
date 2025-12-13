#pragma once

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Parabolic cylinder function D_nu(z)
// Using the formula: D_nu(z) = 2^(nu/2) * exp(-z^2/4) * U(-nu/2, 1/2, z^2/2)
// where U is expressed in terms of 1F1 (Kummer M) via:
// U(a, 1/2, x) = sqrt(pi)/Gamma(a+1/2) * M(a, 1/2, x)
//              - 2*sqrt(pi)/Gamma(a) * sqrt(x) * M(a+1/2, 3/2, x)
template <typename T>
T parabolic_cylinder_d(T nu, T z) {
    T z_sq_half = z * z / T(2);
    T exp_factor = std::exp(-z * z / T(4));
    T pow_factor = std::pow(T(2), nu / T(2));

    // Parameters for U(-nu/2, 1/2, z^2/2)
    T a = -nu / T(2);

    // Compute U(a, 1/2, z^2/2) using Kummer's connection formula
    T sqrt_pi = std::sqrt(boost::math::constants::pi<T>());

    // First term: sqrt(pi)/Gamma(a+1/2) * M(a, 1/2, z^2/2)
    T gamma_a_plus_half = boost::math::tgamma(a + T(0.5));
    T M1 = boost::math::hypergeometric_1F1(a, T(0.5), z_sq_half);
    T term1 = sqrt_pi / gamma_a_plus_half * M1;

    // Second term: -2*sqrt(pi)/Gamma(a) * sqrt(z^2/2) * M(a+1/2, 3/2, z^2/2)
    T gamma_a = boost::math::tgamma(a);
    T sqrt_z_sq_half = std::sqrt(z_sq_half);
    T M2 = boost::math::hypergeometric_1F1(a + T(0.5), T(1.5), z_sq_half);
    T term2 = -T(2) * sqrt_pi / gamma_a * sqrt_z_sq_half * M2;

    T U_val = term1 + term2;

    return pow_factor * exp_factor * U_val;
}

template <typename T>
std::tuple<T, T> parabolic_cylinder_d_backward(T nu, T z) {
    // Use numerical differentiation for gradients
    T h = T(1e-7);

    // Gradient with respect to nu
    T f_nu_plus = parabolic_cylinder_d(nu + h, z);
    T f_nu_minus = parabolic_cylinder_d(nu - h, z);
    T grad_nu = (f_nu_plus - f_nu_minus) / (T(2) * h);

    // Gradient with respect to z
    T f_z_plus = parabolic_cylinder_d(nu, z + h);
    T f_z_minus = parabolic_cylinder_d(nu, z - h);
    T grad_z = (f_z_plus - f_z_minus) / (T(2) * h);

    return std::make_tuple(grad_nu, grad_z);
}

} // namespace torchscience::impl::special_functions
