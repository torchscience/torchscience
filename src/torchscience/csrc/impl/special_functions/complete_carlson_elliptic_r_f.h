#pragma once

#include <boost/math/special_functions/ellint_rf.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T complete_carlson_elliptic_r_f(T x, T y) {
  // Complete Carlson elliptic integral R_F(0, x, y)
  return boost::math::ellint_rf(T(0), x, y);
}

template <typename T>
std::tuple<T, T> complete_carlson_elliptic_r_f_backward(T x, T y) {
  // R_F(0, x, y) partial derivatives
  // Using the relation: R_F(0, x, y) = pi/(2*sqrt(x*y)) * 2F1(1/2, 1/2; 1; 1 - y/x)
  // Partial derivatives can be computed via recurrence relations
  T rf = boost::math::ellint_rf(T(0), x, y);
  T rd_x = boost::math::ellint_rd(T(0), y, x);
  T rd_y = boost::math::ellint_rd(T(0), x, y);

  // dR_F/dx = (R_D(0, y, x) - R_F(0, x, y)) / (2 * x)
  // dR_F/dy = (R_D(0, x, y) - R_F(0, x, y)) / (2 * y)
  T grad_x = (rd_x - rf) / (T(2) * x);
  T grad_y = (rd_y - rf) / (T(2) * y);

  return std::make_tuple(grad_x, grad_y);
}

} // namespace torchscience::impl::special_functions
