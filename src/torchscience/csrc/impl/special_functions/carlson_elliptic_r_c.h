#pragma once

#include <boost/math/special_functions/ellint_rc.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T carlson_elliptic_r_c(T x, T y) {
  return boost::math::ellint_rc(x, y);
}

template <typename T>
std::tuple<T, T> carlson_elliptic_r_c_backward(T x, T y) {
  // R_C(x, y) = (1/2) * integral from 0 to infinity of dt / ((t + x)^(1/2) * (t + y))
  // Partial derivatives:
  // dR_C/dx = -(R_C(x, y) - 1/sqrt(x)) / (2 * (x - y))  when x != y
  // dR_C/dy = (R_C(x, y) - sqrt(x)/y) / (2 * (x - y))   when x != y
  T rc = boost::math::ellint_rc(x, y);
  T sqrt_x = std::sqrt(x);
  T diff = x - y;

  T grad_x, grad_y;
  if (std::abs(diff) < T(1e-10) * std::max(std::abs(x), std::abs(y))) {
    // When x ≈ y, R_C(x, x) = 1/sqrt(x)
    // Use limiting behavior
    grad_x = T(-1) / (T(4) * x * sqrt_x);
    grad_y = T(-1) / (T(4) * y * std::sqrt(y));
  } else {
    grad_x = -(rc - T(1) / sqrt_x) / (T(2) * diff);
    grad_y = (rc - sqrt_x / y) / (T(2) * diff);
  }

  return std::make_tuple(grad_x, grad_y);
}

} // namespace torchscience::impl::special_functions
