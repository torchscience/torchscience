#pragma once

#include <boost/math/special_functions/ellint_rd.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T carlson_elliptic_integral_r_d(T x, T y, T z) {
  // Carlson's elliptic integral R_D(x, y, z)
  // R_D(x, y, z) = (3/2) * integral from 0 to infinity of dt / ((t+x)^(1/2) * (t+y)^(1/2) * (t+z)^(3/2))
  return boost::math::ellint_rd(x, y, z);
}

template <typename T>
std::tuple<T, T, T> carlson_elliptic_integral_r_d_backward(T x, T y, T z) {
  // Partial derivatives of R_D(x, y, z)
  // The derivatives involve R_D and R_F

  T rd = boost::math::ellint_rd(x, y, z);
  T rf = boost::math::ellint_rf(x, y, z);

  T sqrt_x = std::sqrt(x);
  T sqrt_y = std::sqrt(y);
  T sqrt_z = std::sqrt(z);

  // Based on the relations from DLMF 19.18
  // dR_D/dx = -(3/2) * (R_D(x, y, z) - R_D(y, z, x)) / (x - z)  when x != z
  // dR_D/dy = -(3/2) * (R_D(x, y, z) - R_D(x, z, y)) / (y - z)  when y != z
  // dR_D/dz = -(3/2) * [R_D(x, y, z) - 3/(z * sqrt(z)) + 3*sqrt(x*y)/(z*(x+y+z))] / (2*z)

  T grad_x, grad_y, grad_z;

  // For numerical stability, use a uniform approximation
  // dR_D/dx = -3/(2*(x-z)) * [R_D(x,y,z) - R_D(y,z,x)]
  T x_minus_z = x - z;
  T y_minus_z = y - z;

  if (std::abs(x_minus_z) < T(1e-10) * std::max({std::abs(x), std::abs(z), T(1)})) {
    // Limiting case when x ≈ z
    grad_x = T(-3) / (T(4) * z * z) * (rf - T(1) / (T(2) * sqrt_z));
  } else {
    T rd_yzx = boost::math::ellint_rd(y, z, x);
    grad_x = T(-3) / (T(2) * x_minus_z) * (rd - rd_yzx);
  }

  if (std::abs(y_minus_z) < T(1e-10) * std::max({std::abs(y), std::abs(z), T(1)})) {
    // Limiting case when y ≈ z
    grad_y = T(-3) / (T(4) * z * z) * (rf - T(1) / (T(2) * sqrt_z));
  } else {
    T rd_xzy = boost::math::ellint_rd(x, z, y);
    grad_y = T(-3) / (T(2) * y_minus_z) * (rd - rd_xzy);
  }

  // dR_D/dz is more complex
  // Using the homogeneity relation: x*dR_D/dx + y*dR_D/dy + z*dR_D/dz = -(3/2)*R_D
  // z*dR_D/dz = -(3/2)*R_D - x*dR_D/dx - y*dR_D/dy
  grad_z = (T(-3) / T(2) * rd - x * grad_x - y * grad_y) / z;

  return std::make_tuple(grad_x, grad_y, grad_z);
}

} // namespace torchscience::impl::special_functions
