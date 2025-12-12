#pragma once

#include <boost/math/special_functions/chebyshev.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T chebyshev_polynomial_t(T n, T x) {
  // Chebyshev polynomial of the first kind T_n(x)
  // T_n(x) = cos(n * arccos(x)) for |x| <= 1
  return boost::math::chebyshev_t(static_cast<unsigned>(n), x);
}

template <typename T>
std::tuple<T, T> chebyshev_polynomial_t_backward(T n, T x) {
  // Gradient with respect to n is not well-defined (discrete parameter)
  T grad_n = T(0);

  // dT_n/dx = n * U_{n-1}(x) for n >= 1, where U is Chebyshev of second kind
  // For n = 0, dT_0/dx = 0
  T grad_x;
  unsigned int n_int = static_cast<unsigned int>(n);
  if (n_int == 0) {
    grad_x = T(0);
  } else {
    grad_x = n * boost::math::chebyshev_u(n_int - 1, x);
  }

  return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
