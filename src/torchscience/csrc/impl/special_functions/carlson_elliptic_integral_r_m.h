#pragma once

#include <boost/math/special_functions/ellint_rf.hpp>
#include <boost/math/special_functions/ellint_rg.hpp>
#include <boost/math/special_functions/ellint_rd.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T carlson_elliptic_integral_r_m(T x, T y, T z) {
  // R_M(x, y, z) = 2 * R_G(x, y, z) - R_F(x, y, z)
  //
  // This is a useful combination of Carlson symmetric forms that appears
  // in various elliptic integral identities. Named r_m as an alternative
  // naming convention.
  T rg = boost::math::ellint_rg(x, y, z);
  T rf = boost::math::ellint_rf(x, y, z);
  return T(2) * rg - rf;
}

template <typename T>
std::tuple<T, T, T> carlson_elliptic_integral_r_m_backward(T x, T y, T z) {
  // Partial derivatives of R_M(x, y, z) = 2 * R_G(x, y, z) - R_F(x, y, z)
  //
  // dR_M/dx = 2 * dR_G/dx - dR_F/dx
  //
  // From DLMF:
  // dR_F/dx = -R_D(y,z,x) / 6
  // dR_G/dx = (R_F(x,y,z) - x*R_D(y,z,x)/3) / 2
  //
  // So: dR_M/dx = 2 * (R_F - x*R_D(y,z,x)/3) / 2 + R_D(y,z,x) / 6
  //             = R_F - x*R_D(y,z,x)/3 + R_D(y,z,x)/6
  //             = R_F - (2x - 1) * R_D(y,z,x) / 6
  //             = R_F + (1 - 2x) * R_D(y,z,x) / 6

  T rf = boost::math::ellint_rf(x, y, z);
  T rd_yzx = boost::math::ellint_rd(y, z, x);
  T rd_xzy = boost::math::ellint_rd(x, z, y);
  T rd_xyz = boost::math::ellint_rd(x, y, z);

  // dR_G/dx = (R_F - x*R_D(y,z,x)/3) / 2
  // dR_F/dx = -R_D(y,z,x) / 6
  // dR_M/dx = 2 * dR_G/dx - dR_F/dx = R_F - x*R_D(y,z,x)/3 + R_D(y,z,x)/6
  T grad_x = rf - x * rd_yzx / T(3) + rd_yzx / T(6);
  T grad_y = rf - y * rd_xzy / T(3) + rd_xzy / T(6);
  T grad_z = rf - z * rd_xyz / T(3) + rd_xyz / T(6);

  return std::make_tuple(grad_x, grad_y, grad_z);
}

} // namespace torchscience::impl::special_functions
