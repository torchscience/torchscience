#pragma once

#include <boost/math/special_functions/jacobi_theta.hpp>
#include <cmath>
#include <tuple>

template <typename T>
T jacobi_theta_2(T z, T q) {
  return boost::math::jacobi_theta2(z, q);
}

template <typename T>
std::tuple<T, T> jacobi_theta_2_backward(T z, T q) {
  // θ₂(z, q) = 2 * sum_{n=0}^∞ q^((n+1/2)²) * cos((2n+1)z)
  // Use numerical differentiation for robustness
  T eps = std::sqrt(std::numeric_limits<T>::epsilon());

  T theta2_z_plus = boost::math::jacobi_theta2(z + eps, q);
  T theta2_z_minus = boost::math::jacobi_theta2(z - eps, q);
  T grad_z = (theta2_z_plus - theta2_z_minus) / (T(2) * eps);

  T theta2_q_plus = boost::math::jacobi_theta2(z, q + eps);
  T theta2_q_minus = boost::math::jacobi_theta2(z, std::max(q - eps, T(0)));
  T grad_q = (theta2_q_plus - theta2_q_minus) / (T(2) * eps);

  return std::make_tuple(grad_z, grad_q);
}
