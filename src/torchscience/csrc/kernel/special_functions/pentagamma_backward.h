#pragma once

#include "polygamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
C10_HOST_DEVICE T pentagamma_backward(
  T gradient,
  T z
) {
  return gradient * polygamma(T(4), z);
}

template <typename T>
C10_HOST_DEVICE c10::complex<T> pentagamma_backward(
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  return gradient * polygamma(c10::complex<T>(T(4), T(0)), z);
}

} // namespace torchscience::kernel::special_functions
