#pragma once

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/bessel_prime.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T modified_bessel_k(T nu, T x) {
  return boost::math::cyl_bessel_k(nu, x);
}

template <typename T>
std::tuple<T, T> modified_bessel_k_backward(T nu, T x) {
  T grad_nu = T(0);
  T grad_x = boost::math::cyl_bessel_k_prime(nu, x);

  return std::make_tuple(grad_nu, grad_x);
}

} // namespace torchscience::impl::special_functions
