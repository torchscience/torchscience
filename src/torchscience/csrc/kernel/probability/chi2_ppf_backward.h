#pragma once

#include <cmath>
#include <tuple>
#include "chi2_ppf.h"
#include "chi2_pdf.h"

namespace torchscience::kernel::probability {

// Gradient of chi2_ppf with respect to p:
// By implicit differentiation: d(ppf)/dp = 1 / pdf(ppf(p))
template <typename T>
T chi2_ppf_grad_p(T p, T df) {
  if (p <= T(0) || p >= T(1)) return T(0);
  T x = chi2_ppf(p, df);
  T pdf = chi2_pdf(x, df);
  if (pdf < T(1e-15)) return T(0);
  return T(1) / pdf;
}

// Gradient of chi2_ppf with respect to df:
// Uses numerical differentiation
template <typename T>
T chi2_ppf_grad_df(T p, T df) {
  if (p <= T(0) || p >= T(1)) return T(0);
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), std::abs(df));
  T ppf_plus = chi2_ppf(p, df + eps);
  T ppf_minus = chi2_ppf(p, df - eps);
  return (ppf_plus - ppf_minus) / (T(2) * eps);
}

template <typename T>
std::tuple<T, T> chi2_ppf_backward(T grad_output, T p, T df) {
  return {
    grad_output * chi2_ppf_grad_p(p, df),
    grad_output * chi2_ppf_grad_df(p, df)
  };
}

}  // namespace torchscience::kernel::probability
