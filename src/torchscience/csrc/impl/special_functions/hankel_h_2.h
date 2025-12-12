#pragma once

#include <boost/math/special_functions/hankel.hpp>
#include <complex>

namespace torchscience::impl::special_functions {

template <typename T>
std::complex<T> hankel_h_2(std::complex<T> nu, std::complex<T> x) {
  // Hankel function of the second kind: H^(2)_nu(x) = J_nu(x) - i*Y_nu(x)
  // For complex inputs, use the real parts for the Boost function
  return boost::math::cyl_hankel_2(nu.real(), x.real());
}

template <typename T>
T hankel_h_2(T nu, T x) {
  // For real inputs, return the magnitude of the complex result
  auto result = boost::math::cyl_hankel_2(nu, x);
  return std::abs(result);
}

template <typename T>
std::tuple<std::complex<T>, std::complex<T>> hankel_h_2_backward(std::complex<T> nu, std::complex<T> x) {
  // Gradient with respect to nu is not supported
  std::complex<T> grad_nu(T(0), T(0));

  // d H^(2)_nu(x)/dx = (H^(2)_{nu-1}(x) - H^(2)_{nu+1}(x)) / 2
  auto h2_nu_minus_1 = boost::math::cyl_hankel_2(nu.real() - T(1), x.real());
  auto h2_nu_plus_1 = boost::math::cyl_hankel_2(nu.real() + T(1), x.real());
  std::complex<T> grad_x = (h2_nu_minus_1 - h2_nu_plus_1) / T(2);

  return std::make_tuple(grad_nu, grad_x);
}

template <typename T>
std::tuple<T, T> hankel_h_2_backward(T nu, T x) {
  // For real-valued output (magnitude), compute gradient of |H^(2)|
  T grad_nu = T(0);

  auto h2 = boost::math::cyl_hankel_2(nu, x);
  auto h2_nu_minus_1 = boost::math::cyl_hankel_2(nu - T(1), x);
  auto h2_nu_plus_1 = boost::math::cyl_hankel_2(nu + T(1), x);

  // d|H|/dx = Re(H* * dH/dx) / |H|
  auto dh_dx = (h2_nu_minus_1 - h2_nu_plus_1) / T(2);
  T abs_h2 = std::abs(h2);
  if (abs_h2 > T(0)) {
    T grad_x = (h2.real() * dh_dx.real() + h2.imag() * dh_dx.imag()) / abs_h2;
    return std::make_tuple(grad_nu, grad_x);
  }
  return std::make_tuple(grad_nu, T(0));
}

} // namespace torchscience::impl::special_functions
