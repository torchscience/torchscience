#pragma once

#include <tuple>

#include "beta.h"
#include "digamma.h"
#include "trigamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> beta_backward_backward(
    T gg_a,
    T gg_b,
    T gradient,
    T a,
    T b) {
  T beta_ab = beta(a, b);
  T psi_a = digamma(a);
  T psi_b = digamma(b);
  T psi_ab = digamma(a + b);
  T psi1_a = trigamma(a);
  T psi1_b = trigamma(b);
  T psi1_ab = trigamma(a + b);

  T diff_a = psi_a - psi_ab;
  T diff_b = psi_b - psi_ab;

  T gg_output = gg_a * beta_ab * diff_a + gg_b * beta_ab * diff_b;

  return {
    gg_output,
    gg_a * (gradient * beta_ab * (diff_a * diff_a + psi1_a - psi1_ab)) + gg_b * (gradient * beta_ab * (diff_a * diff_b - psi1_ab)),
    gg_a * (gradient * beta_ab * (diff_a * diff_b - psi1_ab)) + gg_b * (gradient * beta_ab * (diff_b * diff_b + psi1_b - psi1_ab))
  };
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> beta_backward_backward(
    c10::complex<T> gg_a,
    c10::complex<T> gg_b,
    c10::complex<T> gradient,
    c10::complex<T> a,
    c10::complex<T> b) {
  c10::complex<T> beta_ab = beta(a, b);
  c10::complex<T> psi_a = digamma(a);
  c10::complex<T> psi_b = digamma(b);
  c10::complex<T> psi_ab = digamma(a + b);
  c10::complex<T> psi1_a = trigamma(a);
  c10::complex<T> psi1_b = trigamma(b);
  c10::complex<T> psi1_ab = trigamma(a + b);

  c10::complex<T> diff_a = psi_a - psi_ab;
  c10::complex<T> diff_b = psi_b - psi_ab;

  c10::complex<T> gg_output = gg_a * beta_ab * diff_a + gg_b * beta_ab * diff_b;

  c10::complex<T> d2_aa = gradient * beta_ab * (diff_a * diff_a + psi1_a - psi1_ab);
  c10::complex<T> d2_ab = gradient * beta_ab * (diff_a * diff_b - psi1_ab);
  c10::complex<T> d2_bb = gradient * beta_ab * (diff_b * diff_b + psi1_b - psi1_ab);

  c10::complex<T> new_grad_a = gg_a * d2_aa + gg_b * d2_ab;
  c10::complex<T> new_grad_b = gg_a * d2_ab + gg_b * d2_bb;

  return {gg_output, new_grad_a, new_grad_b};
}

} // namespace torchscience::kernel::special_functions
