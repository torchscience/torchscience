#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

template <typename T>
inline T inv_sqrt_2pi_pdf() {
  return T(0.3989422804014327);  // 1 / sqrt(2 * pi)
}

// Normal probability density function
// PDF(x; loc, scale) = exp(-(x - loc)^2 / (2 * scale^2)) / (scale * sqrt(2*pi))
template <typename T>
T normal_pdf(T x, T loc, T scale) {
  T z = (x - loc) / scale;
  return inv_sqrt_2pi_pdf<T>() / scale * std::exp(T(-0.5) * z * z);
}

}  // namespace torchscience::kernel::probability
