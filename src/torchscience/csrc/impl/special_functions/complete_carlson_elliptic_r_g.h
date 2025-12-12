#pragma once

#include <boost/math/special_functions/ellint_rg.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T complete_carlson_elliptic_r_g(T x, T y) {
  // Complete Carlson elliptic integral R_G(0, x, y)
  return boost::math::ellint_rg(T(0), x, y);
}

template <typename T>
std::tuple<T, T> complete_carlson_elliptic_r_g_backward(T x, T y) {
  // R_G(0, x, y) partial derivatives
  // R_G(0, x, y) = (y * R_F(0, x, y) + (x - y) * R_D(0, x, y) / 3 + sqrt(x)) / 2
  // Gradients are computed using the relation:
  // dR_G/dx = (R_G(0, x, y) - R_F(0, x, y)) / (2 * x)
  // dR_G/dy = (R_F(0, x, y) - R_G(0, x, y) / y) / 2
  T rg = boost::math::ellint_rg(T(0), x, y);
  T rf = boost::math::ellint_rf(T(0), x, y);

  T grad_x, grad_y;
  if (x < T(1e-10)) {
    // Limiting case when x -> 0
    grad_x = T(0.25) / std::sqrt(y);
    grad_y = T(0.25) * std::sqrt(y) / y;
  } else {
    grad_x = (rg - rf) / (T(2) * x);
    grad_y = (rf - rg / y) / T(2);
  }

  return std::make_tuple(grad_x, grad_y);
}

} // namespace torchscience::impl::special_functions
