#pragma once

#include <tuple>

#include "pentagamma.h"
#include "polygamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
C10_HOST_DEVICE std::tuple<T, T> tetragamma_backward_backward(
  T gradient_gradient,
  T gradient,
  T z
) {
  return {
    gradient_gradient * pentagamma(z),
    gradient_gradient * gradient * polygamma(T(4), z)
  };
}

template <typename T>
C10_HOST_DEVICE std::tuple<c10::complex<T>, c10::complex<T>> tetragamma_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  return {
    gradient_gradient * pentagamma(z),
    gradient_gradient * gradient * polygamma(c10::complex<T>(T(4), T(0)), z)
  };
}

} // namespace torchscience::kernel::special_functions
