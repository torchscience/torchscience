#pragma once

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/bessel_prime.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T spherical_bessel_y(T n, T x) {
  return boost::math::sph_neumann(static_cast<unsigned>(n), x);
}

template <typename T>
std::tuple<T, T> spherical_bessel_y_backward(T n, T x) {
  T grad_n = T(0);
  T grad_x = boost::math::sph_neumann_prime(static_cast<unsigned>(n), x);

  return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
