#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "cos_pi.h"
#include "sin_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> sin_pi_backward_backward(T gg, T g, T x) {
  T x_mod = std::fmod(x, T(2));

  if (x_mod < T(0)) {
    x_mod += T(2);
  }

  T pi = static_cast<T>(M_PI);
  T pi_x = pi * x_mod;

  T sin_pi_x = std::sin(pi_x);
  T cos_pi_x = std::cos(pi_x);

  T gg_output = gg * pi * cos_pi_x;
  T g_x = -gg * g * pi * pi * sin_pi_x;

  return std::make_tuple(gg_output, g_x);
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> sin_pi_backward_backward(
  c10::complex<T> gg,
  c10::complex<T> g,
  c10::complex<T> z
) {
  T pi = static_cast<T>(M_PI);

  c10::complex<T> sin_pi_z = sin_pi(z);
  c10::complex<T> cos_pi_z = cos_pi(z);

  c10::complex<T> gg_output = gg * pi * cos_pi_z;
  c10::complex<T> g_z = -gg * g * pi * pi * sin_pi_z;

  return std::make_tuple(gg_output, g_z);
}

} // namespace torchscience::kernel::special_functions
