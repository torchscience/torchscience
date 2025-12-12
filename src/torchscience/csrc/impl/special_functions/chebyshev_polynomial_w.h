#pragma once

#include <boost/math/special_functions/chebyshev.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T chebyshev_polynomial_w(T n, T x) {
  // Chebyshev polynomial of the fourth kind W_n(x)
  // W_n(x) = sin((n + 1/2) * arccos(x)) / sin(arccos(x)/2) for |x| <= 1
  return boost::math::chebyshev_w(static_cast<unsigned>(n), x);
}

template <typename T>
std::tuple<T, T> chebyshev_polynomial_w_backward(T n, T x) {
  // Gradient with respect to n is not well-defined (discrete parameter)
  T grad_n = T(0);

  // Derivative using numerical approximation for W polynomials
  // dW_n/dx can be computed using recurrence relations
  unsigned int n_int = static_cast<unsigned int>(n);
  T h = T(1e-7);
  T w_plus = boost::math::chebyshev_w(n_int, x + h);
  T w_minus = boost::math::chebyshev_w(n_int, x - h);
  T grad_x = (w_plus - w_minus) / (T(2) * h);

  return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
