#pragma once

#include <boost/math/special_functions/ellint_rg.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include <boost/math/special_functions/ellint_rd.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T carlson_elliptic_integral_r_k(T x, T y, T z) {
  // Carlson's elliptic integral R_G(x, y, z)
  // R_G(x, y, z) = (1/4π) * integral over the unit sphere of sqrt(x*l² + y*m² + z*n²) dΩ
  //
  // Or alternatively:
  // R_G(x, y, z) = (1/4) * integral from 0 to infinity of
  //   [(t+x)(t+y)(t+z)]^(-1/2) * [x/(t+x) + y/(t+y) + z/(t+z)] * t dt
  //
  // Note: Named r_k as an alternative naming convention for R_G.
  return boost::math::ellint_rg(x, y, z);
}

template <typename T>
std::tuple<T, T, T> carlson_elliptic_integral_r_k_backward(T x, T y, T z) {
  // Partial derivatives of R_G(x, y, z)
  // Using the relations from DLMF 19.21:
  // dR_G/dx = (R_F(x,y,z) - x*R_D(y,z,x)/3) / 2
  // dR_G/dy = (R_F(x,y,z) - y*R_D(x,z,y)/3) / 2
  // dR_G/dz = (R_F(x,y,z) - z*R_D(x,y,z)/3) / 2

  T rf = boost::math::ellint_rf(x, y, z);
  T rd_yzx = boost::math::ellint_rd(y, z, x);
  T rd_xzy = boost::math::ellint_rd(x, z, y);
  T rd_xyz = boost::math::ellint_rd(x, y, z);

  T grad_x = (rf - x * rd_yzx / T(3)) / T(2);
  T grad_y = (rf - y * rd_xzy / T(3)) / T(2);
  T grad_z = (rf - z * rd_xyz / T(3)) / T(2);

  return std::make_tuple(grad_x, grad_y, grad_z);
}

} // namespace torchscience::impl::special_functions
