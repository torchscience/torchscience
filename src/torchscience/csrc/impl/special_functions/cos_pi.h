#pragma once

#include <boost/math/special_functions/cos_pi.hpp>
#include <boost/math/special_functions/sin_pi.hpp>
#include <boost/math/constants/constants.hpp>
#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T cos_pi(T x) {
  return boost::math::cos_pi(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> cos_pi(c10::complex<T> z) {
  // cos(π*z) = cos(π*(x+iy)) = cos(πx)cosh(πy) - i*sin(πx)sinh(πy)
  T x = z.real();
  T y = z.imag();
  T pi = boost::math::constants::pi<T>();
  T sin_pi_x = boost::math::sin_pi(x);
  T cos_pi_x = boost::math::cos_pi(x);
  T cosh_pi_y = std::cosh(pi * y);
  T sinh_pi_y = std::sinh(pi * y);
  return c10::complex<T>(cos_pi_x * cosh_pi_y, -sin_pi_x * sinh_pi_y);
}

template <typename T>
C10_HOST_DEVICE T cos_pi_backward(T x) {
  return -boost::math::constants::pi<T>() * boost::math::sin_pi(x);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> cos_pi_backward(c10::complex<T> z) {
  // d/dz cos(π*z) = -π * sin(π*z)
  // sin(π*z) = sin(πx)cosh(πy) + i*cos(πx)sinh(πy)
  T x = z.real();
  T y = z.imag();
  T pi = boost::math::constants::pi<T>();
  T sin_pi_x = boost::math::sin_pi(x);
  T cos_pi_x = boost::math::cos_pi(x);
  T cosh_pi_y = std::cosh(pi * y);
  T sinh_pi_y = std::sinh(pi * y);
  return -pi * c10::complex<T>(sin_pi_x * cosh_pi_y, cos_pi_x * sinh_pi_y);
}

} // namespace torchscience::impl::special_functions
