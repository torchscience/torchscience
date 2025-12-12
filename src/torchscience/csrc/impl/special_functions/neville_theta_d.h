#pragma once

#include <boost/math/special_functions/jacobi_theta.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T neville_theta_d(T k, T u) {
  // θd(k, u) = θ₃(πu/(2K), q) / θ₃(0, q)
  // where K = K(k), q = nome(k)

  if (k <= T(0) || k >= T(1)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  T K_k = boost::math::ellint_1(k);
  T k_prime = std::sqrt(T(1) - k * k);
  T K_k_prime = boost::math::ellint_1(k_prime);

  T q = std::exp(-boost::math::constants::pi<T>() * K_k_prime / K_k);
  T v = boost::math::constants::pi<T>() * u / (T(2) * K_k);

  T theta3_v = boost::math::jacobi_theta3(v, q);
  T theta3_0 = boost::math::jacobi_theta3(T(0), q);

  return theta3_v / theta3_0;
}

template <typename T>
std::tuple<T, T> neville_theta_d_backward(T k, T u) {
  T eps = std::sqrt(std::numeric_limits<T>::epsilon());

  T theta_k_plus = neville_theta_d(std::min(k + eps, T(1) - eps), u);
  T theta_k_minus = neville_theta_d(std::max(k - eps, eps), u);
  T grad_k = (theta_k_plus - theta_k_minus) / (T(2) * eps);

  T theta_u_plus = neville_theta_d(k, u + eps);
  T theta_u_minus = neville_theta_d(k, u - eps);
  T grad_u = (theta_u_plus - theta_u_minus) / (T(2) * eps);

  return std::make_tuple(grad_k, grad_u);
}

} // namespace torchscience::impl::special_functions
