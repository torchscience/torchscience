#pragma once

#include <boost/math/special_functions/bessel.hpp>

template <typename T>
T spherical_bessel_j(T n, T x) {
  return boost::math::sph_bessel(static_cast<unsigned>(n), x);
}

template <typename T>
std::tuple<T, T> spherical_bessel_j_backward(T n, T x) {
  // Gradient with respect to n is not supported (would require
  // derivative of Bessel function with respect to order)
  T grad_n = T(0);

  // Gradient with respect to x:
  // dj_n(x)/dx = j_{n-1}(x) - (n+1)/x * j_n(x)
  unsigned n_int = static_cast<unsigned>(n);
  T j_n = boost::math::sph_bessel(n_int, x);
  T grad_x;
  if (n_int == 0) {
    // dj_0(x)/dx = -j_1(x)
    grad_x = -boost::math::sph_bessel(1, x);
  } else {
    T j_n_minus_1 = boost::math::sph_bessel(n_int - 1, x);
    grad_x = j_n_minus_1 - (n + T(1)) / x * j_n;
  }

  return std::make_tuple(grad_n, grad_x);
}
