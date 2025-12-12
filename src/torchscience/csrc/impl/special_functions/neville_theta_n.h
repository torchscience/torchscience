#pragma once

#include <boost/math/special_functions/jacobi_theta.hpp>
#include <cmath>
#include <tuple>

template <typename T>
T neville_theta_n(T k, T u) {
  return boost::math::neville_theta_n(k, u);
}

template <typename T>
std::tuple<T, T> neville_theta_n_backward(T k, T u) {
  // Use numerical differentiation for robustness
  T eps = std::sqrt(std::numeric_limits<T>::epsilon());

  T theta_k_plus = boost::math::neville_theta_n(k + eps, u);
  T theta_k_minus = boost::math::neville_theta_n(std::max(k - eps, T(0)), u);
  T grad_k = (theta_k_plus - theta_k_minus) / (T(2) * eps);

  T theta_u_plus = boost::math::neville_theta_n(k, u + eps);
  T theta_u_minus = boost::math::neville_theta_n(k, u - eps);
  T grad_u = (theta_u_plus - theta_u_minus) / (T(2) * eps);

  return std::make_tuple(grad_k, grad_u);
}
