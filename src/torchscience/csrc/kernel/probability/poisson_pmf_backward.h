#pragma once

#include <cmath>
#include <tuple>
#include "poisson_pmf.h"

namespace torchscience::kernel::probability {

// Backward for Poisson PMF
// PMF = rate^k * exp(-rate) / k!
// dPMF/drate = PMF * (k/rate - 1)
//
// k is discrete, so its gradient is 0
template <typename T>
std::tuple<T, T> poisson_pmf_backward(T gradient, T k, T rate) {
  k = std::floor(k);

  // Boundary cases
  if (k < T(0) || rate <= T(0)) {
    return {T(0), T(0)};
  }

  T pmf = poisson_pmf(k, rate);

  // dPMF/drate = PMF * (k/rate - 1)
  T grad_rate = gradient * pmf * (k / rate - T(1));

  return {T(0), grad_rate};
}

}  // namespace torchscience::kernel::probability
