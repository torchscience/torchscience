#pragma once

#include <boost/math/special_functions/bessel.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T bessel_j(T nu, T x) {
  return boost::math::cyl_bessel_j(nu, x);
}

template <typename T>
std::tuple<T, T> bessel_j_backward(T nu, T x) {
  // Gradient with respect to nu is not supported (would require
  // derivative of Bessel function with respect to order)
  T grad_nu = T(0);

  // Gradient with respect to x:
  // dJ_nu(x)/dx = (J_{nu-1}(x) - J_{nu+1}(x)) / 2
  T j_nu_minus_1 = boost::math::cyl_bessel_j(nu - T(1), x);
  T j_nu_plus_1 = boost::math::cyl_bessel_j(nu + T(1), x);
  T grad_x = (j_nu_minus_1 - j_nu_plus_1) / T(2);

  return std::make_tuple(grad_nu, grad_x);
}

} // namespace torchscience::impl::special_functions
