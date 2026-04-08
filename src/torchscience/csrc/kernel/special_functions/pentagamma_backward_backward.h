#pragma once

#include <tuple>

#include "polygamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> pentagamma_backward_backward(
  T gradient_gradient,
  T gradient,
  T z
) {
  return {
    gradient_gradient * polygamma(T(4), z),
    gradient_gradient * gradient * polygamma(T(5), z)
  };
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> pentagamma_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  return {
    gradient_gradient * polygamma(c10::complex<T>(T(4), T(0)), z),
    gradient_gradient * gradient * polygamma(c10::complex<T>(T(5), T(0)), z)
  };
}

} // namespace torchscience::kernel::special_functions
