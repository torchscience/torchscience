#pragma once

#include <cmath>
#include <tuple>
#include "binomial_pmf.h"

namespace torchscience::kernel::probability {

// Backward for binomial PMF
// PMF = C(n, k) * p^k * (1-p)^(n-k)
// dPMF/dp = PMF * (k/p - (n-k)/(1-p))
//
// k and n are discrete, so their gradients are 0
template <typename T>
std::tuple<T, T, T> binomial_pmf_backward(T gradient, T k, T n, T p) {
  k = std::floor(k);

  // Boundary cases
  if (k < T(0) || k > n) {
    return {T(0), T(0), T(0)};
  }

  // Edge cases for p
  if (p <= T(0) || p >= T(1)) {
    return {T(0), T(0), T(0)};
  }

  T pmf = binomial_pmf(k, n, p);

  // dPMF/dp = PMF * (k/p - (n-k)/(1-p))
  T grad_p = gradient * pmf * (k / p - (n - k) / (T(1) - p));

  return {T(0), T(0), grad_p};
}

}  // namespace torchscience::kernel::probability
