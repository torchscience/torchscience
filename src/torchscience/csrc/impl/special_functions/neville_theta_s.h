#pragma once

#include <boost/math/special_functions/jacobi_theta.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T neville_theta_s(T k, T u) {
  // θs(k, u) = θ₁(πu/(2K), q) / (θ₁'(0, q) * πu/(2K))
  // where K = K(k), q = nome(k)

  if (k <= T(0) || k >= T(1)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  T K_k = boost::math::ellint_1(k);
  T k_prime = std::sqrt(T(1) - k * k);
  T K_k_prime = boost::math::ellint_1(k_prime);

  T q = std::exp(-boost::math::constants::pi<T>() * K_k_prime / K_k);
  T v = boost::math::constants::pi<T>() * u / (T(2) * K_k);

  if (std::abs(v) < std::numeric_limits<T>::epsilon()) {
    return T(1);  // θs(k, 0) = 1 by definition
  }

  T theta1_v = boost::math::jacobi_theta1(v, q);

  // θ₁'(0, q) = π/2 * θ₂(0,q) * θ₃(0,q) * θ₄(0,q)
  T theta2_0 = boost::math::jacobi_theta2(T(0), q);
  T theta3_0 = boost::math::jacobi_theta3(T(0), q);
  T theta4_0 = boost::math::jacobi_theta4(T(0), q);
  T theta1_prime_0 = boost::math::constants::pi<T>() * theta2_0 * theta3_0 * theta4_0 / T(2);

  return theta1_v / (v * theta1_prime_0);
}

template <typename T>
std::tuple<T, T> neville_theta_s_backward(T k, T u) {
  T eps = std::sqrt(std::numeric_limits<T>::epsilon());

  T theta_k_plus = neville_theta_s(std::min(k + eps, T(1) - eps), u);
  T theta_k_minus = neville_theta_s(std::max(k - eps, eps), u);
  T grad_k = (theta_k_plus - theta_k_minus) / (T(2) * eps);

  T theta_u_plus = neville_theta_s(k, u + eps);
  T theta_u_minus = neville_theta_s(k, u - eps);
  T grad_u = (theta_u_plus - theta_u_minus) / (T(2) * eps);

  return std::make_tuple(grad_k, grad_u);
}

} // namespace torchscience::impl::special_functions
