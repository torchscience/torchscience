#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "cosh_pi.h"
#include "sinh_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> sinh_pi_backward_backward(T gg, T g, T x) {
  T pi = static_cast<T>(M_PI);
  T pi_x = pi * x;

  T sinh_pi_x = std::sinh(pi_x);
  T cosh_pi_x = std::cosh(pi_x);

  T gg_output = gg * pi * cosh_pi_x;
  T g_x = gg * g * pi * pi * sinh_pi_x;

  return std::make_tuple(gg_output, g_x);
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> sinh_pi_backward_backward(
  c10::complex<T> gg,
  c10::complex<T> g,
  c10::complex<T> z
) {
  T pi = static_cast<T>(M_PI);

  c10::complex<T> sinh_pi_z = sinh_pi(z);
  c10::complex<T> cosh_pi_z = cosh_pi(z);

  c10::complex<T> gg_output = gg * pi * cosh_pi_z;
  c10::complex<T> g_z = gg * g * pi * pi * sinh_pi_z;

  return std::make_tuple(gg_output, g_z);
}

} // namespace torchscience::kernel::special_functions
