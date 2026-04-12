#pragma once

#include <cmath>
#include <tuple>

#include "chebyshev_polynomial_v.h"
#include "chebyshev_polynomial_v_backward.h"
#include "chebyshev_polynomial_w.h"
#include "chebyshev_polynomial_w_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
C10_HOST_DEVICE std::tuple<T, T, T> chebyshev_polynomial_w_backward_backward(
  T gradient_gradient_x,
  T gradient_gradient_n,
  T gradient,
  T x,
  T n
) {
  if (n < T(1)) {
    return {T(0), T(0), T(0)};
  }

  if (std::abs(x + T(1)) < T(1e-10)) {
    // Boundary at x = -1
    T sign;

    if (static_cast<int>(n) % 2 == 0) {
      sign = T(1);
    } else {
      sign = T(-1);
    }

    // First derivative at x = -1
    T first_deriv = sign * (T(2) * n + T(1)) * n * (n + T(1)) / T(3);

    // Second derivative at x = -1
    T second_deriv = sign * (T(2) * n + T(1)) * n * (n + T(1)) * (n - T(1)) * (n + T(2)) / T(15);

    T gradient_gradient_output = gradient_gradient_x * first_deriv;
    T new_grad_x = gradient_gradient_x * gradient * second_deriv;

    return {
      gradient_gradient_output,
      new_grad_x,
      T(0)
    };
  }

  // General case: dW_n/dx = ((2n+1)*V_n(x) - W_n(x)) / (2*(x+1))
  // Second derivative via quotient rule: d/dx[num/denom]

  T v_n = chebyshev_polynomial_v(x, n);
  T w_n = chebyshev_polynomial_w(x, n);
  T two_n_plus_1 = T(2) * n + T(1);
  T denom = T(2) * (x + T(1));

  // Derivatives of numerator components
  T v_backward_x = std::get<0>(chebyshev_polynomial_v_backward(T(1), x, n));
  T w_backward_x = std::get<0>(chebyshev_polynomial_w_backward(T(1), x, n));

  T numerator = two_n_plus_1 * v_n - w_n;
  T d_numerator = two_n_plus_1 * v_backward_x - w_backward_x;
  T d_denom = T(2);

  T first_deriv = numerator / denom;
  T second_deriv = (d_numerator * denom - numerator * d_denom) / (denom * denom);

  T gradient_gradient_output = gradient_gradient_x * first_deriv;
  T new_grad_x = gradient_gradient_x * gradient * second_deriv;

  return {
    gradient_gradient_output,
    new_grad_x,
    T(0)
  };
}

} // namespace torchscience::kernel::special_functions
