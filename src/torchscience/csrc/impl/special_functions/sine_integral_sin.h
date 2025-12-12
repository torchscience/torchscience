#pragma once

#include <torchscience/csrc/impl/special_functions/sine_integral_si.h>

namespace torchscience::impl::special_functions {

template <typename T>
C10_HOST_DEVICE T sine_integral_sin(T x) {
  // si(x) = Si(x) - π/2 = -integral from x to infinity of sin(t)/t dt
  // This is the "shifted" sine integral that goes to 0 as x -> infinity
  T pi_2 = T(3.14159265358979323846264338327950288) / T(2);
  return sine_integral_si(x) - pi_2;
}

template <typename T>
C10_HOST_DEVICE T sine_integral_sin_backward(T x) {
  // d/dx si(x) = d/dx Si(x) = sin(x)/x = sinc(x) (unnormalized)
  if (x == T(0)) {
    return T(1);
  }
  return std::sin(x) / x;
}

} // namespace torchscience::impl::special_functions
