#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "spherical_harmonic_y.h"
#include "spherical_harmonic_y_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for spherical harmonic Y_l^m(theta, phi)
//
// Uses finite differences on the backward function for second derivatives.

template <typename T>
C10_HOST_DEVICE std::tuple<T, T, T, T, T> spherical_harmonic_y_backward_backward(
    T gg_l, T gg_m, T gg_theta, T gg_phi,
    T gradient, T l, T m, T theta, T phi) {

  T eps = T(1e-5);
  T gradient_gradient_output = T(0);
  T new_grad_l = T(0);
  T new_grad_m = T(0);
  T new_grad_theta = T(0);
  T new_grad_phi = T(0);

  // Contribution from gg_l
  if (std::abs(gg_l) > T(1e-15)) {
    auto [gl, gm, gtheta, gphi] = spherical_harmonic_y_backward(T(1), l, m, theta, phi);
    gradient_gradient_output += gg_l * gl;

    auto [gl_plus, gm_plus, gtheta_plus, gphi_plus] = spherical_harmonic_y_backward(gradient, l + eps, m, theta, phi);
    auto [gl_minus, gm_minus, gtheta_minus, gphi_minus] = spherical_harmonic_y_backward(gradient, l - eps, m, theta, phi);
    new_grad_l += gg_l * (gl_plus - gl_minus) / (T(2) * eps);

    auto [gl_m_plus, gm_m_plus, gtheta_m_plus, gphi_m_plus] = spherical_harmonic_y_backward(gradient, l, m + eps, theta, phi);
    auto [gl_m_minus, gm_m_minus, gtheta_m_minus, gphi_m_minus] = spherical_harmonic_y_backward(gradient, l, m - eps, theta, phi);
    new_grad_m += gg_l * (gl_m_plus - gl_m_minus) / (T(2) * eps);

    auto [gl_t_plus, gm_t_plus, gtheta_t_plus, gphi_t_plus] = spherical_harmonic_y_backward(gradient, l, m, theta + eps, phi);
    auto [gl_t_minus, gm_t_minus, gtheta_t_minus, gphi_t_minus] = spherical_harmonic_y_backward(gradient, l, m, theta - eps, phi);
    new_grad_theta += gg_l * (gl_t_plus - gl_t_minus) / (T(2) * eps);

    auto [gl_p_plus, gm_p_plus, gtheta_p_plus, gphi_p_plus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi + eps);
    auto [gl_p_minus, gm_p_minus, gtheta_p_minus, gphi_p_minus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi - eps);
    new_grad_phi += gg_l * (gl_p_plus - gl_p_minus) / (T(2) * eps);
  }

  // Contribution from gg_m
  if (std::abs(gg_m) > T(1e-15)) {
    auto [gl, gm, gtheta, gphi] = spherical_harmonic_y_backward(T(1), l, m, theta, phi);
    gradient_gradient_output += gg_m * gm;

    auto [gl_plus, gm_plus, gtheta_plus, gphi_plus] = spherical_harmonic_y_backward(gradient, l + eps, m, theta, phi);
    auto [gl_minus, gm_minus, gtheta_minus, gphi_minus] = spherical_harmonic_y_backward(gradient, l - eps, m, theta, phi);
    new_grad_l += gg_m * (gm_plus - gm_minus) / (T(2) * eps);

    auto [gl_m_plus, gm_m_plus, gtheta_m_plus, gphi_m_plus] = spherical_harmonic_y_backward(gradient, l, m + eps, theta, phi);
    auto [gl_m_minus, gm_m_minus, gtheta_m_minus, gphi_m_minus] = spherical_harmonic_y_backward(gradient, l, m - eps, theta, phi);
    new_grad_m += gg_m * (gm_m_plus - gm_m_minus) / (T(2) * eps);

    auto [gl_t_plus, gm_t_plus, gtheta_t_plus, gphi_t_plus] = spherical_harmonic_y_backward(gradient, l, m, theta + eps, phi);
    auto [gl_t_minus, gm_t_minus, gtheta_t_minus, gphi_t_minus] = spherical_harmonic_y_backward(gradient, l, m, theta - eps, phi);
    new_grad_theta += gg_m * (gm_t_plus - gm_t_minus) / (T(2) * eps);

    auto [gl_p_plus, gm_p_plus, gtheta_p_plus, gphi_p_plus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi + eps);
    auto [gl_p_minus, gm_p_minus, gtheta_p_minus, gphi_p_minus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi - eps);
    new_grad_phi += gg_m * (gm_p_plus - gm_p_minus) / (T(2) * eps);
  }

  // Contribution from gg_theta
  if (std::abs(gg_theta) > T(1e-15)) {
    auto [gl, gm, gtheta, gphi] = spherical_harmonic_y_backward(T(1), l, m, theta, phi);
    gradient_gradient_output += gg_theta * gtheta;

    auto [gl_plus, gm_plus, gtheta_plus, gphi_plus] = spherical_harmonic_y_backward(gradient, l + eps, m, theta, phi);
    auto [gl_minus, gm_minus, gtheta_minus, gphi_minus] = spherical_harmonic_y_backward(gradient, l - eps, m, theta, phi);
    new_grad_l += gg_theta * (gtheta_plus - gtheta_minus) / (T(2) * eps);

    auto [gl_m_plus, gm_m_plus, gtheta_m_plus, gphi_m_plus] = spherical_harmonic_y_backward(gradient, l, m + eps, theta, phi);
    auto [gl_m_minus, gm_m_minus, gtheta_m_minus, gphi_m_minus] = spherical_harmonic_y_backward(gradient, l, m - eps, theta, phi);
    new_grad_m += gg_theta * (gtheta_m_plus - gtheta_m_minus) / (T(2) * eps);

    auto [gl_t_plus, gm_t_plus, gtheta_t_plus, gphi_t_plus] = spherical_harmonic_y_backward(gradient, l, m, theta + eps, phi);
    auto [gl_t_minus, gm_t_minus, gtheta_t_minus, gphi_t_minus] = spherical_harmonic_y_backward(gradient, l, m, theta - eps, phi);
    new_grad_theta += gg_theta * (gtheta_t_plus - gtheta_t_minus) / (T(2) * eps);

    auto [gl_p_plus, gm_p_plus, gtheta_p_plus, gphi_p_plus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi + eps);
    auto [gl_p_minus, gm_p_minus, gtheta_p_minus, gphi_p_minus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi - eps);
    new_grad_phi += gg_theta * (gphi_p_plus - gphi_p_minus) / (T(2) * eps);
  }

  // Contribution from gg_phi
  if (std::abs(gg_phi) > T(1e-15)) {
    auto [gl, gm, gtheta, gphi] = spherical_harmonic_y_backward(T(1), l, m, theta, phi);
    gradient_gradient_output += gg_phi * gphi;

    auto [gl_plus, gm_plus, gtheta_plus, gphi_plus] = spherical_harmonic_y_backward(gradient, l + eps, m, theta, phi);
    auto [gl_minus, gm_minus, gtheta_minus, gphi_minus] = spherical_harmonic_y_backward(gradient, l - eps, m, theta, phi);
    new_grad_l += gg_phi * (gphi_plus - gphi_minus) / (T(2) * eps);

    auto [gl_m_plus, gm_m_plus, gtheta_m_plus, gphi_m_plus] = spherical_harmonic_y_backward(gradient, l, m + eps, theta, phi);
    auto [gl_m_minus, gm_m_minus, gtheta_m_minus, gphi_m_minus] = spherical_harmonic_y_backward(gradient, l, m - eps, theta, phi);
    new_grad_m += gg_phi * (gphi_m_plus - gphi_m_minus) / (T(2) * eps);

    auto [gl_t_plus, gm_t_plus, gtheta_t_plus, gphi_t_plus] = spherical_harmonic_y_backward(gradient, l, m, theta + eps, phi);
    auto [gl_t_minus, gm_t_minus, gtheta_t_minus, gphi_t_minus] = spherical_harmonic_y_backward(gradient, l, m, theta - eps, phi);
    new_grad_theta += gg_phi * (gphi_t_plus - gphi_t_minus) / (T(2) * eps);

    auto [gl_p_plus, gm_p_plus, gtheta_p_plus, gphi_p_plus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi + eps);
    auto [gl_p_minus, gm_p_minus, gtheta_p_minus, gphi_p_minus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi - eps);
    new_grad_phi += gg_phi * (gphi_p_plus - gphi_p_minus) / (T(2) * eps);
  }

  return {gradient_gradient_output, new_grad_l, new_grad_m, new_grad_theta, new_grad_phi};
}

// Complex version
template <typename T>
C10_HOST_DEVICE std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
spherical_harmonic_y_backward_backward(
    c10::complex<T> gg_l, c10::complex<T> gg_m, c10::complex<T> gg_theta, c10::complex<T> gg_phi,
    c10::complex<T> gradient, c10::complex<T> l, c10::complex<T> m,
    c10::complex<T> theta, c10::complex<T> phi) {

  T eps_val = T(1e-5);
  c10::complex<T> eps(eps_val, T(0));
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  c10::complex<T> gradient_gradient_output = zero;
  c10::complex<T> new_grad_l = zero;
  c10::complex<T> new_grad_m = zero;
  c10::complex<T> new_grad_theta = zero;
  c10::complex<T> new_grad_phi = zero;

  // Contribution from gg_l
  if (std::abs(gg_l) > eps_val) {
    auto [gl, gm, gtheta, gphi] = spherical_harmonic_y_backward(one, l, m, theta, phi);
    gradient_gradient_output += gg_l * std::conj(gl);

    auto [gl_plus, gm_plus, gtheta_plus, gphi_plus] = spherical_harmonic_y_backward(gradient, l + eps, m, theta, phi);
    auto [gl_minus, gm_minus, gtheta_minus, gphi_minus] = spherical_harmonic_y_backward(gradient, l - eps, m, theta, phi);
    new_grad_l += gg_l * std::conj((gl_plus - gl_minus) / (two * eps));

    auto [gl_m_plus, gm_m_plus, gtheta_m_plus, gphi_m_plus] = spherical_harmonic_y_backward(gradient, l, m + eps, theta, phi);
    auto [gl_m_minus, gm_m_minus, gtheta_m_minus, gphi_m_minus] = spherical_harmonic_y_backward(gradient, l, m - eps, theta, phi);
    new_grad_m += gg_l * std::conj((gl_m_plus - gl_m_minus) / (two * eps));

    auto [gl_t_plus, gm_t_plus, gtheta_t_plus, gphi_t_plus] = spherical_harmonic_y_backward(gradient, l, m, theta + eps, phi);
    auto [gl_t_minus, gm_t_minus, gtheta_t_minus, gphi_t_minus] = spherical_harmonic_y_backward(gradient, l, m, theta - eps, phi);
    new_grad_theta += gg_l * std::conj((gl_t_plus - gl_t_minus) / (two * eps));

    auto [gl_p_plus, gm_p_plus, gtheta_p_plus, gphi_p_plus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi + eps);
    auto [gl_p_minus, gm_p_minus, gtheta_p_minus, gphi_p_minus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi - eps);
    new_grad_phi += gg_l * std::conj((gl_p_plus - gl_p_minus) / (two * eps));
  }

  // Contribution from gg_m
  if (std::abs(gg_m) > eps_val) {
    auto [gl, gm, gtheta, gphi] = spherical_harmonic_y_backward(one, l, m, theta, phi);
    gradient_gradient_output += gg_m * std::conj(gm);

    auto [gl_plus, gm_plus, gtheta_plus, gphi_plus] = spherical_harmonic_y_backward(gradient, l + eps, m, theta, phi);
    auto [gl_minus, gm_minus, gtheta_minus, gphi_minus] = spherical_harmonic_y_backward(gradient, l - eps, m, theta, phi);
    new_grad_l += gg_m * std::conj((gm_plus - gm_minus) / (two * eps));

    auto [gl_m_plus, gm_m_plus, gtheta_m_plus, gphi_m_plus] = spherical_harmonic_y_backward(gradient, l, m + eps, theta, phi);
    auto [gl_m_minus, gm_m_minus, gtheta_m_minus, gphi_m_minus] = spherical_harmonic_y_backward(gradient, l, m - eps, theta, phi);
    new_grad_m += gg_m * std::conj((gm_m_plus - gm_m_minus) / (two * eps));

    auto [gl_t_plus, gm_t_plus, gtheta_t_plus, gphi_t_plus] = spherical_harmonic_y_backward(gradient, l, m, theta + eps, phi);
    auto [gl_t_minus, gm_t_minus, gtheta_t_minus, gphi_t_minus] = spherical_harmonic_y_backward(gradient, l, m, theta - eps, phi);
    new_grad_theta += gg_m * std::conj((gm_t_plus - gm_t_minus) / (two * eps));

    auto [gl_p_plus, gm_p_plus, gtheta_p_plus, gphi_p_plus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi + eps);
    auto [gl_p_minus, gm_p_minus, gtheta_p_minus, gphi_p_minus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi - eps);
    new_grad_phi += gg_m * std::conj((gm_p_plus - gm_p_minus) / (two * eps));
  }

  // Contribution from gg_theta
  if (std::abs(gg_theta) > eps_val) {
    auto [gl, gm, gtheta, gphi] = spherical_harmonic_y_backward(one, l, m, theta, phi);
    gradient_gradient_output += gg_theta * std::conj(gtheta);

    auto [gl_plus, gm_plus, gtheta_plus, gphi_plus] = spherical_harmonic_y_backward(gradient, l + eps, m, theta, phi);
    auto [gl_minus, gm_minus, gtheta_minus, gphi_minus] = spherical_harmonic_y_backward(gradient, l - eps, m, theta, phi);
    new_grad_l += gg_theta * std::conj((gtheta_plus - gtheta_minus) / (two * eps));

    auto [gl_m_plus, gm_m_plus, gtheta_m_plus, gphi_m_plus] = spherical_harmonic_y_backward(gradient, l, m + eps, theta, phi);
    auto [gl_m_minus, gm_m_minus, gtheta_m_minus, gphi_m_minus] = spherical_harmonic_y_backward(gradient, l, m - eps, theta, phi);
    new_grad_m += gg_theta * std::conj((gtheta_m_plus - gtheta_m_minus) / (two * eps));

    auto [gl_t_plus, gm_t_plus, gtheta_t_plus, gphi_t_plus] = spherical_harmonic_y_backward(gradient, l, m, theta + eps, phi);
    auto [gl_t_minus, gm_t_minus, gtheta_t_minus, gphi_t_minus] = spherical_harmonic_y_backward(gradient, l, m, theta - eps, phi);
    new_grad_theta += gg_theta * std::conj((gtheta_t_plus - gtheta_t_minus) / (two * eps));

    auto [gl_p_plus, gm_p_plus, gtheta_p_plus, gphi_p_plus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi + eps);
    auto [gl_p_minus, gm_p_minus, gtheta_p_minus, gphi_p_minus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi - eps);
    new_grad_phi += gg_theta * std::conj((gphi_p_plus - gphi_p_minus) / (two * eps));
  }

  // Contribution from gg_phi
  if (std::abs(gg_phi) > eps_val) {
    auto [gl, gm, gtheta, gphi] = spherical_harmonic_y_backward(one, l, m, theta, phi);
    gradient_gradient_output += gg_phi * std::conj(gphi);

    auto [gl_plus, gm_plus, gtheta_plus, gphi_plus] = spherical_harmonic_y_backward(gradient, l + eps, m, theta, phi);
    auto [gl_minus, gm_minus, gtheta_minus, gphi_minus] = spherical_harmonic_y_backward(gradient, l - eps, m, theta, phi);
    new_grad_l += gg_phi * std::conj((gphi_plus - gphi_minus) / (two * eps));

    auto [gl_m_plus, gm_m_plus, gtheta_m_plus, gphi_m_plus] = spherical_harmonic_y_backward(gradient, l, m + eps, theta, phi);
    auto [gl_m_minus, gm_m_minus, gtheta_m_minus, gphi_m_minus] = spherical_harmonic_y_backward(gradient, l, m - eps, theta, phi);
    new_grad_m += gg_phi * std::conj((gphi_m_plus - gphi_m_minus) / (two * eps));

    auto [gl_t_plus, gm_t_plus, gtheta_t_plus, gphi_t_plus] = spherical_harmonic_y_backward(gradient, l, m, theta + eps, phi);
    auto [gl_t_minus, gm_t_minus, gtheta_t_minus, gphi_t_minus] = spherical_harmonic_y_backward(gradient, l, m, theta - eps, phi);
    new_grad_theta += gg_phi * std::conj((gphi_t_plus - gphi_t_minus) / (two * eps));

    auto [gl_p_plus, gm_p_plus, gtheta_p_plus, gphi_p_plus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi + eps);
    auto [gl_p_minus, gm_p_minus, gtheta_p_minus, gphi_p_minus] = spherical_harmonic_y_backward(gradient, l, m, theta, phi - eps);
    new_grad_phi += gg_phi * std::conj((gphi_p_plus - gphi_p_minus) / (two * eps));
  }

  return {gradient_gradient_output, new_grad_l, new_grad_m, new_grad_theta, new_grad_phi};
}

} // namespace torchscience::kernel::special_functions
