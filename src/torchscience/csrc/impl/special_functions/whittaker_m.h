#pragma once

#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Whittaker M function M_{kappa, mu}(z)
// M_{kappa, mu}(z) = z^(mu + 1/2) * exp(-z/2) * 1F1(mu - kappa + 1/2, 2*mu + 1, z)
template <typename T>
T whittaker_m(T kappa, T mu, T z) {
    T a = mu - kappa + T(0.5);
    T b = T(2) * mu + T(1);

    T z_power = std::pow(z, mu + T(0.5));
    T exp_factor = std::exp(-z / T(2));
    T hypergeom = boost::math::hypergeometric_1F1(a, b, z);

    return z_power * exp_factor * hypergeom;
}

template <typename T>
std::tuple<T, T, T> whittaker_m_backward(T kappa, T mu, T z) {
    // Use numerical differentiation for gradients
    T h = T(1e-7);

    // Gradient with respect to kappa
    T f_kappa_plus = whittaker_m(kappa + h, mu, z);
    T f_kappa_minus = whittaker_m(kappa - h, mu, z);
    T grad_kappa = (f_kappa_plus - f_kappa_minus) / (T(2) * h);

    // Gradient with respect to mu
    T f_mu_plus = whittaker_m(kappa, mu + h, z);
    T f_mu_minus = whittaker_m(kappa, mu - h, z);
    T grad_mu = (f_mu_plus - f_mu_minus) / (T(2) * h);

    // Gradient with respect to z
    T f_z_plus = whittaker_m(kappa, mu, z + h);
    T f_z_minus = whittaker_m(kappa, mu, z - h);
    T grad_z = (f_z_plus - f_z_minus) / (T(2) * h);

    return std::make_tuple(grad_kappa, grad_mu, grad_z);
}

} // namespace torchscience::impl::special_functions
