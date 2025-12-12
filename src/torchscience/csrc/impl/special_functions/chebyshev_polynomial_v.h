#pragma once

#include <boost/math/special_functions/chebyshev.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T chebyshev_polynomial_v(T n, T x) {
  // Chebyshev polynomial of the third kind V_n(x)
  // V_n(x) = cos((n + 1/2) * arccos(x)) / cos(arccos(x)/2) for |x| <= 1
  return boost::math::chebyshev_v(static_cast<unsigned>(n), x);
}

template <typename T>
std::tuple<T, T> chebyshev_polynomial_v_backward(T n, T x) {
  // Gradient with respect to n is not well-defined (discrete parameter)
  T grad_n = T(0);

  // Derivative using numerical approximation for V polynomials
  // dV_n/dx can be computed using recurrence relations
  unsigned int n_int = static_cast<unsigned int>(n);
  T h = T(1e-7);
  T v_plus = boost::math::chebyshev_v(n_int, x + h);
  T v_minus = boost::math::chebyshev_v(n_int, x - h);
  T grad_x = (v_plus - v_minus) / (T(2) * h);

  return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
