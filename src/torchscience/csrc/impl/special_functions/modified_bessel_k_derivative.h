#pragma once

#include <boost/math/special_functions/bessel_prime.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T modified_bessel_k_derivative(T nu, T x) {
  return boost::math::cyl_bessel_k_prime(nu, x);
}

template <typename T>
std::tuple<T, T> modified_bessel_k_derivative_backward(T nu, T x) {
  T d_x = -(boost::math::cyl_bessel_k_prime(nu - T(1), x) +
            boost::math::cyl_bessel_k_prime(nu + T(1), x)) / T(2);
  return std::make_tuple(T(0), d_x);
}

} // namespace torchscience::impl::special_functions
