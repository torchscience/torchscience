#pragma once

#include <boost/math/special_functions/jacobi_theta.hpp>
#include <cmath>
#include <tuple>

template <typename T>
T jacobi_theta_4(T z, T q) {
  return boost::math::jacobi_theta4(z, q);
}

template <typename T>
std::tuple<T, T> jacobi_theta_4_backward(T z, T q) {
  // θ₄(z, q) = 1 + 2 * sum_{n=1}^∞ (-1)^n * q^(n²) * cos(2nz)
  // Use numerical differentiation for robustness
  T eps = std::sqrt(std::numeric_limits<T>::epsilon());

  T theta4_z_plus = boost::math::jacobi_theta4(z + eps, q);
  T theta4_z_minus = boost::math::jacobi_theta4(z - eps, q);
  T grad_z = (theta4_z_plus - theta4_z_minus) / (T(2) * eps);

  T theta4_q_plus = boost::math::jacobi_theta4(z, q + eps);
  T theta4_q_minus = boost::math::jacobi_theta4(z, std::max(q - eps, T(0)));
  T grad_q = (theta4_q_plus - theta4_q_minus) / (T(2) * eps);

  return std::make_tuple(grad_z, grad_q);
}
