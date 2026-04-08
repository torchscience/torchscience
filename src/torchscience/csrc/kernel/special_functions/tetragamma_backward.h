#pragma once

#include "pentagamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T tetragamma_backward(
  T gradient,
  T z
) {
  return gradient * pentagamma(z);
}

template <typename T>
c10::complex<T> tetragamma_backward(
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  return gradient * pentagamma(z);
}

} // namespace torchscience::kernel::special_functions
