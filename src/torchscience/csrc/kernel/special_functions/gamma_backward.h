#pragma once

#include "digamma.h"
#include "gamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T gamma_backward(T gradient, T z) {
  return gradient * gamma(z) * digamma(z);
}

} // namespace torchscience::kernel::special_functions
