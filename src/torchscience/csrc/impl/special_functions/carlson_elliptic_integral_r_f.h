#pragma once

#include <boost/math/special_functions/ellint_rf.hpp>
#include <boost/math/special_functions/ellint_rd.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T carlson_elliptic_integral_r_f(T x, T y, T z) {
  // Carlson's elliptic integral R_F(x, y, z)
  // R_F(x, y, z) = (1/2) * integral from 0 to infinity of dt / ((t+x)^(1/2) * (t+y)^(1/2) * (t+z)^(1/2))
  return boost::math::ellint_rf(x, y, z);
}

template <typename T>
std::tuple<T, T, T> carlson_elliptic_integral_r_f_backward(T x, T y, T z) {
  // Partial derivatives of R_F(x, y, z)
  // Using the relations from DLMF 19.18:
  // dR_F/dx = -R_D(y, z, x) / 6
  // dR_F/dy = -R_D(x, z, y) / 6
  // dR_F/dz = -R_D(x, y, z) / 6

  T rd_yzx = boost::math::ellint_rd(y, z, x);
  T rd_xzy = boost::math::ellint_rd(x, z, y);
  T rd_xyz = boost::math::ellint_rd(x, y, z);

  T grad_x = -rd_yzx / T(6);
  T grad_y = -rd_xzy / T(6);
  T grad_z = -rd_xyz / T(6);

  return std::make_tuple(grad_x, grad_y, grad_z);
}

} // namespace torchscience::impl::special_functions
