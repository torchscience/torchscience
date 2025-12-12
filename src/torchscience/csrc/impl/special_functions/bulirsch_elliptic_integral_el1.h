#pragma once

#include <boost/math/special_functions/ellint_rf.hpp>
#include <cmath>

template <typename T>
T bulirsch_elliptic_integral_el1(T x, T kc) {
  // Bulirsch's incomplete elliptic integral of the first kind
  // el1(x, kc) = x * R_F(1, 1 + kc^2 * x^2, 1 + x^2)
  // where kc is the complementary modulus
  T x2 = x * x;
  T kc2 = kc * kc;
  return x * boost::math::ellint_rf(T(1), T(1) + kc2 * x2, T(1) + x2);
}

template <typename T>
std::tuple<T, T> bulirsch_elliptic_integral_el1_backward(T x, T kc) {
  // Gradient computation is complex for elliptic integrals
  // d/dx el1(x, kc) = 1 / sqrt((1 + x^2)(1 + kc^2 * x^2))
  T x2 = x * x;
  T kc2 = kc * kc;
  T grad_x = T(1) / std::sqrt((T(1) + x2) * (T(1) + kc2 * x2));

  // Gradient with respect to kc is more complex, set to zero for now
  T grad_kc = T(0);

  return std::make_tuple(grad_x, grad_kc);
}
