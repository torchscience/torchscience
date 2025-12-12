#pragma once

#include <boost/math/special_functions/bessel.hpp>

template <typename T>
T spherical_bessel_y(T n, T x) {
  return boost::math::sph_neumann(static_cast<unsigned>(n), x);
}

template <typename T>
std::tuple<T, T> spherical_bessel_y_backward(T n, T x) {
  // Gradient with respect to n is not supported (would require
  // derivative of Bessel function with respect to order)
  T grad_n = T(0);

  // Gradient with respect to x:
  // dy_n(x)/dx = y_{n-1}(x) - (n+1)/x * y_n(x)
  unsigned n_int = static_cast<unsigned>(n);
  T y_n = boost::math::sph_neumann(n_int, x);
  T grad_x;
  if (n_int == 0) {
    // dy_0(x)/dx = -y_1(x)
    grad_x = -boost::math::sph_neumann(1, x);
  } else {
    T y_n_minus_1 = boost::math::sph_neumann(n_int - 1, x);
    grad_x = y_n_minus_1 - (n + T(1)) / x * y_n;
  }

  return std::make_tuple(grad_n, grad_x);
}
