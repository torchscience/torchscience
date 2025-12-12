#pragma once

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/constants/constants.hpp>
#include <cmath>

template <typename T>
T spherical_modified_bessel_i(T n, T x) {
  // i_n(x) = sqrt(pi/(2x)) * I_{n+1/2}(x)
  if (x == T(0)) {
    return n == T(0) ? T(1) : T(0);
  }
  T nu = n + T(0.5);
  T i_nu = boost::math::cyl_bessel_i(nu, x);
  return std::sqrt(boost::math::constants::pi<T>() / (T(2) * x)) * i_nu;
}

template <typename T>
std::tuple<T, T> spherical_modified_bessel_i_backward(T n, T x) {
  // Gradient with respect to n is not supported
  T grad_n = T(0);

  // Gradient with respect to x:
  // di_n(x)/dx = i_{n-1}(x) - (n+1)/x * i_n(x)
  unsigned n_int = static_cast<unsigned>(n);
  T i_n = spherical_modified_bessel_i(n, x);
  T grad_x;
  if (x == T(0)) {
    grad_x = T(0);
  } else if (n_int == 0) {
    // di_0(x)/dx = i_{-1}(x) - 1/x * i_0(x) = cosh(x)/x - sinh(x)/x^2
    // which equals i_1(x) when using the recurrence relation differently
    // Using: di_0/dx = (cosh(x) - sinh(x)/x) / x
    grad_x = std::cosh(x) / x - std::sinh(x) / (x * x);
  } else {
    T i_n_minus_1 = spherical_modified_bessel_i(n - T(1), x);
    grad_x = i_n_minus_1 - (n + T(1)) / x * i_n;
  }

  return std::make_tuple(grad_n, grad_x);
}
