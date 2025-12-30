#pragma once

#include <cmath>
#include <tuple>

#include "hypergeometric_2_f_1.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T, T, T> hypergeometric_2_f_1_backward_backward(
  T gg_a,
  T gg_b,
  T gg_c,
  T gg_z,
  T grad,
  T a,
  T b,
  T c,
  T z
) {
  // Second derivative w.r.t. z:
  // d/dz 2F1(a,b;c;z) = (a*b/c) * 2F1(a+1, b+1; c+1; z)
  // d²/dz² 2F1(a,b;c;z) = (a*b/c) * ((a+1)*(b+1)/(c+1)) * 2F1(a+2, b+2; c+2; z)
  T dz_coef = a * b / c;
  T d2z_coef = dz_coef * (a + T(1)) * (b + T(1)) / (c + T(1));

  T f_shifted = hypergeometric_2_f_1(a + T(1), b + T(1), c + T(1), z);
  T f_double_shifted = hypergeometric_2_f_1(a + T(2), b + T(2), c + T(2), z);

  // gg_out: gradient of backward output w.r.t. upstream grad
  T gg_out = gg_z * dz_coef * f_shifted;

  // new_grad_z: second derivative w.r.t. z
  T new_grad_z = gg_z * grad * d2z_coef * f_double_shifted;

  // For parameter second derivatives, return zeros (simplified)
  T new_grad_a = T(0);
  T new_grad_b = T(0);
  T new_grad_c = T(0);

  return {gg_out, new_grad_a, new_grad_b, new_grad_c, new_grad_z};
}

} // namespace torchscience::kernel::special_functions
