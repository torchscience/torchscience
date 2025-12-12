#pragma once

#include <boost/math/special_functions/bessel.hpp>

template <typename T>
T modified_bessel_i(T nu, T x) {
  return boost::math::cyl_bessel_i(nu, x);
}

template <typename T>
std::tuple<T, T> modified_bessel_i_backward(T nu, T x) {
  // Gradient with respect to nu is not supported (would require
  // derivative of Bessel function with respect to order)
  T grad_nu = T(0);

  // Gradient with respect to x:
  // dI_nu(x)/dx = (I_{nu-1}(x) + I_{nu+1}(x)) / 2
  T i_nu_minus_1 = boost::math::cyl_bessel_i(nu - T(1), x);
  T i_nu_plus_1 = boost::math::cyl_bessel_i(nu + T(1), x);
  T grad_x = (i_nu_minus_1 + i_nu_plus_1) / T(2);

  return std::make_tuple(grad_nu, grad_x);
}
