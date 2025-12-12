#pragma once

#include <boost/math/special_functions/hankel.hpp>
#include <complex>

template <typename T>
std::complex<T> hankel_h_1(std::complex<T> nu, std::complex<T> x) {
  // Hankel function of the first kind: H^(1)_nu(x) = J_nu(x) + i*Y_nu(x)
  // For complex inputs, use the real parts for the Boost function
  return boost::math::cyl_hankel_1(nu.real(), x.real());
}

template <typename T>
T hankel_h_1(T nu, T x) {
  // For real inputs, return the magnitude of the complex result
  auto result = boost::math::cyl_hankel_1(nu, x);
  return std::abs(result);
}

template <typename T>
std::tuple<std::complex<T>, std::complex<T>> hankel_h_1_backward(std::complex<T> nu, std::complex<T> x) {
  // Gradient with respect to nu is not supported
  std::complex<T> grad_nu(T(0), T(0));

  // d H^(1)_nu(x)/dx = (H^(1)_{nu-1}(x) - H^(1)_{nu+1}(x)) / 2
  auto h1_nu_minus_1 = boost::math::cyl_hankel_1(nu.real() - T(1), x.real());
  auto h1_nu_plus_1 = boost::math::cyl_hankel_1(nu.real() + T(1), x.real());
  std::complex<T> grad_x = (h1_nu_minus_1 - h1_nu_plus_1) / T(2);

  return std::make_tuple(grad_nu, grad_x);
}

template <typename T>
std::tuple<T, T> hankel_h_1_backward(T nu, T x) {
  // For real-valued output (magnitude), compute gradient of |H^(1)|
  T grad_nu = T(0);

  auto h1 = boost::math::cyl_hankel_1(nu, x);
  auto h1_nu_minus_1 = boost::math::cyl_hankel_1(nu - T(1), x);
  auto h1_nu_plus_1 = boost::math::cyl_hankel_1(nu + T(1), x);

  // d|H|/dx = Re(H* * dH/dx) / |H|
  auto dh_dx = (h1_nu_minus_1 - h1_nu_plus_1) / T(2);
  T abs_h1 = std::abs(h1);
  if (abs_h1 > T(0)) {
    T grad_x = (h1.real() * dh_dx.real() + h1.imag() * dh_dx.imag()) / abs_h1;
    return std::make_tuple(grad_nu, grad_x);
  }
  return std::make_tuple(grad_nu, T(0));
}
