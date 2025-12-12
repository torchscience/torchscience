#pragma once

#include <boost/math/special_functions/jacobi_theta.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T jacobi_theta_1(T z, T q) {
  return boost::math::jacobi_theta1(z, q);
}

template <typename T>
std::tuple<T, T> jacobi_theta_1_backward(T z, T q) {
  // θ₁(z, q) = 2 * sum_{n=0}^∞ (-1)^n * q^((n+1/2)²) * sin((2n+1)z)
  // ∂θ₁/∂z = 2 * sum_{n=0}^∞ (-1)^n * q^((n+1/2)²) * (2n+1) * cos((2n+1)z)
  // Use numerical differentiation for robustness
  T eps = std::sqrt(std::numeric_limits<T>::epsilon());

  T theta1 = boost::math::jacobi_theta1(z, q);
  T theta1_z_plus = boost::math::jacobi_theta1(z + eps, q);
  T theta1_z_minus = boost::math::jacobi_theta1(z - eps, q);
  T grad_z = (theta1_z_plus - theta1_z_minus) / (T(2) * eps);

  T theta1_q_plus = boost::math::jacobi_theta1(z, q + eps);
  T theta1_q_minus = boost::math::jacobi_theta1(z, std::max(q - eps, T(0)));
  T grad_q = (theta1_q_plus - theta1_q_minus) / (T(2) * eps);

  return std::make_tuple(grad_z, grad_q);
}

} // namespace torchscience::impl::special_functions
