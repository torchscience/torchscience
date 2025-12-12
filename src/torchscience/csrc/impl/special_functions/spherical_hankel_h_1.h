#pragma once

#include <boost/math/special_functions/hankel.hpp>
#include <complex>

namespace torchscience::impl::special_functions {

template <typename T>
std::complex<T> spherical_hankel_h_1(std::complex<T> n, std::complex<T> x) {
  // Spherical Hankel function of the first kind: h^(1)_n(x) = j_n(x) + i*y_n(x)
  // For complex inputs, use the real parts for the Boost function
  return boost::math::sph_hankel_1(static_cast<unsigned>(n.real()), x.real());
}

template <typename T>
T spherical_hankel_h_1(T n, T x) {
  // For real inputs, return the magnitude of the complex result
  auto result = boost::math::sph_hankel_1(static_cast<unsigned>(n), x);
  return std::abs(result);
}

template <typename T>
std::tuple<std::complex<T>, std::complex<T>> spherical_hankel_h_1_backward(std::complex<T> n, std::complex<T> x) {
  // Gradient with respect to n is not supported (discrete parameter)
  std::complex<T> grad_n(T(0), T(0));

  // d h^(1)_n(x)/dx = h^(1)_{n-1}(x) - (n+1)/x * h^(1)_n(x)
  unsigned n_int = static_cast<unsigned>(n.real());
  auto h1_n = boost::math::sph_hankel_1(n_int, x.real());
  std::complex<T> grad_x;
  if (n_int > 0) {
    auto h1_n_minus_1 = boost::math::sph_hankel_1(n_int - 1, x.real());
    grad_x = h1_n_minus_1 - std::complex<T>(T(n_int + 1), T(0)) / x * h1_n;
  } else {
    // For n=0: d h^(1)_0(x)/dx = -h^(1)_1(x)
    auto h1_1 = boost::math::sph_hankel_1(1u, x.real());
    grad_x = -h1_1;
  }

  return std::make_tuple(grad_n, grad_x);
}

template <typename T>
std::tuple<T, T> spherical_hankel_h_1_backward(T n, T x) {
  // For real-valued output (magnitude), compute gradient of |h^(1)|
  T grad_n = T(0);

  unsigned n_int = static_cast<unsigned>(n);
  auto h1_n = boost::math::sph_hankel_1(n_int, x);

  std::complex<T> dh_dx;
  if (n_int > 0) {
    auto h1_n_minus_1 = boost::math::sph_hankel_1(n_int - 1, x);
    dh_dx = h1_n_minus_1 - std::complex<T>(T(n_int + 1), T(0)) / std::complex<T>(x, T(0)) * h1_n;
  } else {
    auto h1_1 = boost::math::sph_hankel_1(1u, x);
    dh_dx = -h1_1;
  }

  // d|H|/dx = Re(H* * dH/dx) / |H|
  T abs_h1 = std::abs(h1_n);
  if (abs_h1 > T(0)) {
    T grad_x = (h1_n.real() * dh_dx.real() + h1_n.imag() * dh_dx.imag()) / abs_h1;
    return std::make_tuple(grad_n, grad_x);
  }
  return std::make_tuple(grad_n, T(0));
}

} // namespace torchscience::impl::special_functions
