#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "cos_pi.h"
#include "tan_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> tan_pi_backward_backward(T gg, T g, T x) {
  T pi = static_cast<T>(M_PI);

  T cos_pi_x = cos_pi(x);
  T tan_pi_x = tan_pi(x);

  T sec2_pi_x = T(1) / (cos_pi_x * cos_pi_x);

  T gg_output = gg * pi * sec2_pi_x;
  T g_x = gg * g * T(2) * pi * pi * tan_pi_x * sec2_pi_x;

  return std::make_tuple(gg_output, g_x);
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> tan_pi_backward_backward(
  c10::complex<T> gg,
  c10::complex<T> g,
  c10::complex<T> z
) {
  T pi = static_cast<T>(M_PI);

  c10::complex<T> cos_pi_z = cos_pi(z);
  c10::complex<T> tan_pi_z = tan_pi(z);

  c10::complex<T> sec2_pi_z = c10::complex<T>(T(1), T(0)) / (cos_pi_z * cos_pi_z);

  c10::complex<T> gg_output = gg * pi * sec2_pi_z;
  c10::complex<T> g_z = gg * g * T(2) * pi * pi * tan_pi_z * sec2_pi_z;

  return std::make_tuple(gg_output, g_z);
}

} // namespace torchscience::kernel::special_functions
