#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "cosh_pi.h"
#include "sinh_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> cosh_pi_backward_backward(T gradient_gradient, T g, T x) {
  T pi = static_cast<T>(M_PI);
  T pi_x = pi * x;

  return std::make_tuple(gradient_gradient * pi * std::sinh(pi_x), gradient_gradient * g * pi * pi * std::cosh(pi_x));
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> cosh_pi_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> g,
  c10::complex<T> z
) {
  T pi = static_cast<T>(M_PI);

  return std::make_tuple(gradient_gradient * pi * sinh_pi(z), gradient_gradient * g * pi * pi * cosh_pi(z));
}

} // namespace torchscience::kernel::special_functions
