#pragma once

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Tricomi confluent hypergeometric function U(a, b, z)
// For non-integer b:
// U(a, b, z) = [Gamma(1-b)/Gamma(a-b+1)] * M(a, b, z)
//            + [Gamma(b-1)/Gamma(a)] * z^(1-b) * M(a-b+1, 2-b, z)
template <typename T>
T tricomi_u(T a, T b, T z) {
    T gamma_1_minus_b = boost::math::tgamma(T(1) - b);
    T gamma_a_minus_b_plus_1 = boost::math::tgamma(a - b + T(1));
    T gamma_b_minus_1 = boost::math::tgamma(b - T(1));
    T gamma_a = boost::math::tgamma(a);

    T M1 = boost::math::hypergeometric_1F1(a, b, z);
    T M2 = boost::math::hypergeometric_1F1(a - b + T(1), T(2) - b, z);

    T term1 = (gamma_1_minus_b / gamma_a_minus_b_plus_1) * M1;
    T term2 = (gamma_b_minus_1 / gamma_a) * std::pow(z, T(1) - b) * M2;

    return term1 + term2;
}

// Whittaker W function W_{kappa, mu}(z)
// W_{kappa, mu}(z) = z^(mu + 1/2) * exp(-z/2) * U(mu - kappa + 1/2, 2*mu + 1, z)
template <typename T>
T whittaker_w(T kappa, T mu, T z) {
    T a = mu - kappa + T(0.5);
    T b = T(2) * mu + T(1);

    T z_power = std::pow(z, mu + T(0.5));
    T exp_factor = std::exp(-z / T(2));
    T U_val = tricomi_u(a, b, z);

    return z_power * exp_factor * U_val;
}

template <typename T>
std::tuple<T, T, T> whittaker_w_backward(T kappa, T mu, T z) {
    // Use numerical differentiation for gradients
    T h = T(1e-7);

    // Gradient with respect to kappa
    T f_kappa_plus = whittaker_w(kappa + h, mu, z);
    T f_kappa_minus = whittaker_w(kappa - h, mu, z);
    T grad_kappa = (f_kappa_plus - f_kappa_minus) / (T(2) * h);

    // Gradient with respect to mu
    T f_mu_plus = whittaker_w(kappa, mu + h, z);
    T f_mu_minus = whittaker_w(kappa, mu - h, z);
    T grad_mu = (f_mu_plus - f_mu_minus) / (T(2) * h);

    // Gradient with respect to z
    T f_z_plus = whittaker_w(kappa, mu, z + h);
    T f_z_minus = whittaker_w(kappa, mu, z - h);
    T grad_z = (f_z_plus - f_z_minus) / (T(2) * h);

    return std::make_tuple(grad_kappa, grad_mu, grad_z);
}

} // namespace torchscience::impl::special_functions
