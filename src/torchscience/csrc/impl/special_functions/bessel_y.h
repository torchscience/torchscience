#pragma once

#include <boost/math/special_functions/bessel.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T bessel_y(T nu, T x) {
  return boost::math::cyl_neumann(nu, x);
}

template <typename T>
std::tuple<T, T> bessel_y_backward(T nu, T x) {
  // Gradient with respect to nu is not supported (would require
  // derivative of Bessel function with respect to order)
  T grad_nu = T(0);

  // Gradient with respect to x:
  // dY_nu(x)/dx = (Y_{nu-1}(x) - Y_{nu+1}(x)) / 2
  T y_nu_minus_1 = boost::math::cyl_neumann(nu - T(1), x);
  T y_nu_plus_1 = boost::math::cyl_neumann(nu + T(1), x);
  T grad_x = (y_nu_minus_1 - y_nu_plus_1) / T(2);

  return std::make_tuple(grad_nu, grad_x);
}

} // namespace torchscience::impl::special_functions
