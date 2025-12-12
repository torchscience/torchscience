#pragma once

#include <boost/math/special_functions/chebyshev.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T chebyshev_polynomial_u(T n, T x) {
  // Chebyshev polynomial of the second kind U_n(x)
  // U_n(x) = sin((n+1) * arccos(x)) / sin(arccos(x)) for |x| <= 1
  return boost::math::chebyshev_u(static_cast<unsigned>(n), x);
}

template <typename T>
std::tuple<T, T> chebyshev_polynomial_u_backward(T n, T x) {
  // Gradient with respect to n is not well-defined (discrete parameter)
  T grad_n = T(0);

  // dU_n/dx = ((n+1) * T_{n+1}(x) - x * U_n(x)) / (x^2 - 1) for |x| != 1
  // Using the recurrence relation derivative
  unsigned int n_int = static_cast<unsigned int>(n);
  T x2_minus_1 = x * x - T(1);

  T grad_x;
  if (std::abs(x2_minus_1) < T(1e-10)) {
    // At x = +/- 1, use limit
    grad_x = (n + T(1)) * (n + T(2)) * (n + T(1)) / T(3);
    if (x < T(0) && (n_int % 2 == 0)) {
      grad_x = -grad_x;
    }
  } else {
    T t_np1 = boost::math::chebyshev_t(n_int + 1, x);
    T u_n = boost::math::chebyshev_u(n_int, x);
    grad_x = ((n + T(1)) * t_np1 - x * u_n) / x2_minus_1;
  }

  return std::make_tuple(grad_n, grad_x);
}

} // namespace torchscience::impl::special_functions
