#pragma once

#include <tuple>

#include "digamma.h"
#include "gamma.h"
#include "trigamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> gamma_backward_backward(
  T gradient_gradient,
  T gradient,
  T z
) {
  T gamma_z = gamma(z);

  T psi = digamma(z);

  return {
    gradient_gradient * gamma_z * psi,
    gradient_gradient * gradient * gamma_z * (psi * psi + trigamma(z))
  };
}

} // namespace torchscience::kernel::special_functions
