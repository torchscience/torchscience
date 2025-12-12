#pragma once

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/constants/constants.hpp>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
T spherical_modified_bessel_k(T n, T x) {
  // k_n(x) = sqrt(pi/(2x)) * K_{n+1/2}(x)
  if (x == T(0)) {
    return std::numeric_limits<T>::infinity();
  }
  T nu = n + T(0.5);
  T k_nu = boost::math::cyl_bessel_k(nu, x);
  return std::sqrt(boost::math::constants::pi<T>() / (T(2) * x)) * k_nu;
}

template <typename T>
std::tuple<T, T> spherical_modified_bessel_k_backward(T n, T x) {
  // Gradient with respect to n is not supported
  T grad_n = T(0);

  // Gradient with respect to x:
  // dk_n(x)/dx = -k_{n-1}(x) - (n+1)/x * k_n(x)
  unsigned n_int = static_cast<unsigned>(n);
  T k_n = spherical_modified_bessel_k(n, x);
  T grad_x;
  if (x == T(0)) {
    grad_x = -std::numeric_limits<T>::infinity();
  } else if (n_int == 0) {
    // dk_0(x)/dx = -k_{-1}(x) - 1/x * k_0(x)
    // k_{-1}(x) = k_1(x) for spherical Bessel functions
    T k_1 = spherical_modified_bessel_k(T(1), x);
    grad_x = -k_1 - k_n / x;
  } else {
    T k_n_minus_1 = spherical_modified_bessel_k(n - T(1), x);
    grad_x = -k_n_minus_1 - (n + T(1)) / x * k_n;
  }

  return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
