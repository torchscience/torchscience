#pragma once

#include <boost/math/special_functions/ellint_rj.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include <boost/math/special_functions/ellint_rc.hpp>
#include <boost/math/special_functions/ellint_rd.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T carlson_elliptic_integral_r_j(T x, T y, T z, T p) {
  // Carlson's elliptic integral R_J(x, y, z, p)
  // R_J(x, y, z, p) = (3/2) * integral from 0 to infinity of
  //   dt / ((t+p) * sqrt((t+x)(t+y)(t+z)))
  return boost::math::ellint_rj(x, y, z, p);
}

template <typename T>
std::tuple<T, T, T, T> carlson_elliptic_integral_r_j_backward(T x, T y, T z, T p) {
  // Partial derivatives of R_J(x, y, z, p)
  // Using the homogeneity relation and symmetry properties

  T rj = boost::math::ellint_rj(x, y, z, p);
  T rf = boost::math::ellint_rf(x, y, z);
  T rd_xyz = boost::math::ellint_rd(x, y, z);
  T rd_yzx = boost::math::ellint_rd(y, z, x);
  T rd_zxy = boost::math::ellint_rd(z, x, y);

  // From DLMF 19.18, the derivatives involve R_D and R_J itself
  // dR_J/dx = -(3/2) * (R_J(x,y,z,p) - R_J(y,z,p,x)) / (2*(x-p)) for distinct args
  // More generally, we use numerical differentiation for stability

  T eps = T(1e-7) * std::max({std::abs(x), std::abs(y), std::abs(z), std::abs(p), T(1)});

  T grad_x, grad_y, grad_z, grad_p;

  // Use the relation: dR_J/dx = -3 * R_D(y,z,p,x) / 6 for x, y, z symmetric
  // But R_J has 4 args, need more careful treatment

  // Simplified: use homogeneity x*dRJ/dx + y*dRJ/dy + z*dRJ/dz + p*dRJ/dp = -3/2 * R_J
  // And symmetry in x, y, z

  // Using partial derivative formula from Carlson's original work:
  // dR_J/dx = -3/(2*(x-p)) * [R_J - 3/(x-p) * (R_C(ab/cd, p*(x+y+z+p)/((x-p)^2 + ...)) - R_F)]
  // This is complex, use a simpler approximation based on R_D

  // Approximation using the fact that R_D(x,y,z) = R_J(x,y,z,z)
  // And the derivative relations
  T x_minus_p = x - p;
  T y_minus_p = y - p;
  T z_minus_p = z - p;

  // For numerical stability, use finite differences when args are close
  if (std::abs(x_minus_p) < eps) {
    T rj_plus = boost::math::ellint_rj(x + eps, y, z, p);
    T rj_minus = boost::math::ellint_rj(x - eps, y, z, p);
    grad_x = (rj_plus - rj_minus) / (T(2) * eps);
  } else {
    T rj_yzpx = boost::math::ellint_rj(y, z, p, x);
    grad_x = T(-3) / (T(2) * x_minus_p) * (rj - rj_yzpx);
  }

  if (std::abs(y_minus_p) < eps) {
    T rj_plus = boost::math::ellint_rj(x, y + eps, z, p);
    T rj_minus = boost::math::ellint_rj(x, y - eps, z, p);
    grad_y = (rj_plus - rj_minus) / (T(2) * eps);
  } else {
    T rj_xzpy = boost::math::ellint_rj(x, z, p, y);
    grad_y = T(-3) / (T(2) * y_minus_p) * (rj - rj_xzpy);
  }

  if (std::abs(z_minus_p) < eps) {
    T rj_plus = boost::math::ellint_rj(x, y, z + eps, p);
    T rj_minus = boost::math::ellint_rj(x, y, z - eps, p);
    grad_z = (rj_plus - rj_minus) / (T(2) * eps);
  } else {
    T rj_xypz = boost::math::ellint_rj(x, y, p, z);
    grad_z = T(-3) / (T(2) * z_minus_p) * (rj - rj_xypz);
  }

  // Use homogeneity relation for grad_p
  // x*grad_x + y*grad_y + z*grad_z + p*grad_p = -3/2 * R_J
  grad_p = (T(-3) / T(2) * rj - x * grad_x - y * grad_y - z * grad_z) / p;

  return std::make_tuple(grad_x, grad_y, grad_z, grad_p);
}

} // namespace torchscience::impl::special_functions
