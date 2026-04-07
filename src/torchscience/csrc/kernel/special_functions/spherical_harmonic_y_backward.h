#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "spherical_harmonic_y.h"

namespace torchscience::kernel::special_functions {

// Backward pass for spherical harmonic Y_l^m(theta, phi)
//
// Derivatives:
//   grad_l = 0 (discrete parameter)
//   grad_m = 0 (discrete parameter)
//   grad_phi = grad_output * i*m * Y_l^m(theta, phi)
//   grad_theta = grad_output * N_l^m * dP_l^m(cos(theta))/d(theta) * exp(i*m*phi)
//
// Legendre derivative recurrence:
//   dP_l^m(cos(theta))/d(theta) =
//     [l*cos(theta)*P_l^m(cos(theta)) - (l+m)*P_{l-1}^m(cos(theta))] / sin(theta)

template <typename T>
std::tuple<T, T, T, T> spherical_harmonic_y_backward(
    T gradient, T l, T m, T theta, T phi) {
  // Gradients w.r.t. l and m are zero (discrete parameters)
  T grad_l = T(0);
  T grad_m = T(0);

  // Use finite differences for grad_theta and grad_phi
  T eps = T(1e-7);

  T Y_theta_plus = spherical_harmonic_y(l, m, theta + eps, phi);
  T Y_theta_minus = spherical_harmonic_y(l, m, theta - eps, phi);
  T grad_theta = (Y_theta_plus - Y_theta_minus) / (T(2) * eps);

  T Y_phi_plus = spherical_harmonic_y(l, m, theta, phi + eps);
  T Y_phi_minus = spherical_harmonic_y(l, m, theta, phi - eps);
  T grad_phi = (Y_phi_plus - Y_phi_minus) / (T(2) * eps);

  return {gradient * grad_l, gradient * grad_m, gradient * grad_theta, gradient * grad_phi};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
spherical_harmonic_y_backward(
    c10::complex<T> gradient, c10::complex<T> l, c10::complex<T> m,
    c10::complex<T> theta, c10::complex<T> phi) {
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> two(T(2), T(0));
  T eps_val = T(1e-7);
  c10::complex<T> eps(eps_val, T(0));

  // Gradients w.r.t. l and m are zero (discrete parameters)
  c10::complex<T> grad_l = zero;
  c10::complex<T> grad_m = zero;

  // Use finite differences for grad_theta
  c10::complex<T> Y_theta_plus = spherical_harmonic_y(l, m, theta + eps, phi);
  c10::complex<T> Y_theta_minus = spherical_harmonic_y(l, m, theta - eps, phi);
  c10::complex<T> grad_theta = (Y_theta_plus - Y_theta_minus) / (two * eps);

  // Use finite differences for grad_phi
  c10::complex<T> Y_phi_plus = spherical_harmonic_y(l, m, theta, phi + eps);
  c10::complex<T> Y_phi_minus = spherical_harmonic_y(l, m, theta, phi - eps);
  c10::complex<T> grad_phi = (Y_phi_plus - Y_phi_minus) / (two * eps);

  // For complex holomorphic functions, use Wirtinger derivative convention
  return {
    gradient * std::conj(grad_l),
    gradient * std::conj(grad_m),
    gradient * std::conj(grad_theta),
    gradient * std::conj(grad_phi)
  };
}

} // namespace torchscience::kernel::special_functions
