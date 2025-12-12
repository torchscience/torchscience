#pragma once

#include <boost/math/special_functions/expint.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T exponential_integral_e(T n, T x) {
  return boost::math::expint(static_cast<unsigned>(n), x);
}

template <typename T>
std::tuple<T, T> exponential_integral_e_backward(T n, T x) {
  // Gradient with respect to n is not supported
  T grad_n = T(0);

  // Gradient with respect to x:
  // dE_n(x)/dx = -E_{n-1}(x)
  unsigned n_int = static_cast<unsigned>(n);
  T grad_x;
  if (n_int == 0) {
    // dE_0(x)/dx = -E_{-1}(x) = -e^{-x}/x
    grad_x = -std::exp(-x) / x;
  } else {
    grad_x = -boost::math::expint(n_int - 1, x);
  }

  return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
