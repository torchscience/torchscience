#pragma once

#include <cmath>
#include <tuple>
#include "chi2_sf.h"
#include "chi2_cdf_backward.h"

namespace torchscience::kernel::probability {

// SF = 1 - CDF, so gradients are negated
// d(SF)/dx = -d(CDF)/dx
// d(SF)/ddf = -d(CDF)/ddf

template <typename T>
std::tuple<T, T> chi2_sf_backward(T grad_output, T x, T df) {
  auto [grad_x, grad_df] = chi2_cdf_backward(grad_output, x, df);
  return {-grad_x, -grad_df};
}

}  // namespace torchscience::kernel::probability
