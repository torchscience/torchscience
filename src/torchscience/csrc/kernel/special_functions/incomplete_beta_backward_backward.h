#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "digamma.h"
#include "incomplete_beta.h"
#include "incomplete_beta_backward.h"
#include "log_gamma.h"
#include "trigamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
T log_squared_weighted_beta_integral(T x, T a, T b, int weight_type) {
  if (x <= T(0)) return T(0);

  T log_beta_val = log_gamma(a) + log_gamma(b) - log_gamma(a + b);
  T beta_val = std::exp(log_beta_val);

  T eps = gauss_kronrod_eps<T>();
  T tol = eps * T(10);
  int max_depth = 7;

  T lower = eps;
  T upper = std::min(x, T(1) - eps);

  if (lower >= upper) {
    return T(0);
  }

  auto integrand = [a, b, weight_type](T t) -> T {
    if (t <= T(0) || t >= T(1)) return T(0);

    T log_integrand = (a - T(1)) * std::log(t) + (b - T(1)) * std::log(T(1) - t);
    T base_val = std::exp(log_integrand);

    T log_t = std::log(t);
    T log_1mt = std::log(T(1) - t);

    switch (weight_type) {
      case 0: return log_t * log_t * base_val;
      case 1: return log_1mt * log_1mt * base_val;
      case 2: return log_t * log_1mt * base_val;
      default: return T(0);
    }
  };

  T integral = adaptive_integrate(integrand, lower, upper, tol, max_depth);

  return integral / beta_val;
}

} // namespace detail

template <typename T>
std::tuple<T, T, T, T> incomplete_beta_backward_backward(
    T gg_x, T gg_a, T gg_b,
    T gradient, T x, T a, T b) {

  T gg_out = T(0);
  T new_grad_x = T(0);
  T new_grad_a = T(0);
  T new_grad_b = T(0);

  if (x <= T(0) || x >= T(1)) {
    return {gg_out, new_grad_x, new_grad_a, new_grad_b};
  }

  T log_beta_val = detail::log_beta(a, b);
  T beta_val = std::exp(log_beta_val);

  T log_pdf = (a - T(1)) * std::log(x) + (b - T(1)) * std::log(T(1) - x) - log_beta_val;
  T pdf = std::exp(log_pdf);

  T I_x = incomplete_beta(x, a, b);

  T psi_a = digamma(a);
  T psi_b = digamma(b);
  T psi_ab = digamma(a + b);
  T psi1_a = trigamma(a);
  T psi1_b = trigamma(b);
  T psi1_ab = trigamma(a + b);

  // gg_x contributions
  gg_out += gg_x * pdf;

  T d_pdf_dx = pdf * ((a - T(1)) / x - (b - T(1)) / (T(1) - x));
  new_grad_x += gg_x * gradient * d_pdf_dx;

  T d_pdf_da = pdf * (std::log(x) - (psi_a - psi_ab));
  new_grad_a += gg_x * gradient * d_pdf_da;

  T d_pdf_db = pdf * (std::log(T(1) - x) - (psi_b - psi_ab));
  new_grad_b += gg_x * gradient * d_pdf_db;

  // gg_a contributions
  T log_int_a = detail::log_weighted_beta_integral(x, a, b, true);
  T grad_a_over_g = log_int_a - I_x * (psi_a - psi_ab);

  gg_out += gg_a * grad_a_over_g;

  new_grad_x += gg_a * gradient * std::log(x) * pdf;

  T log_sq_int_aa = detail::log_squared_weighted_beta_integral(x, a, b, 0);
  T d_grad_a_da = log_sq_int_aa
                - T(2) * log_int_a * (psi_a - psi_ab)
                + I_x * (psi_a - psi_ab) * (psi_a - psi_ab)
                - I_x * (psi1_a - psi1_ab);
  new_grad_a += gg_a * gradient * d_grad_a_da;

  T log_sq_int_ab = detail::log_squared_weighted_beta_integral(x, a, b, 2);
  T log_int_b = detail::log_weighted_beta_integral(x, a, b, false);
  T d_grad_a_db = log_sq_int_ab
                - log_int_a * (psi_b - psi_ab)
                - log_int_b * (psi_a - psi_ab)
                + I_x * (psi_a - psi_ab) * (psi_b - psi_ab)
                + I_x * psi1_ab;
  new_grad_b += gg_a * gradient * d_grad_a_db;

  // gg_b contributions
  T grad_b_over_g = log_int_b - I_x * (psi_b - psi_ab);

  gg_out += gg_b * grad_b_over_g;

  new_grad_x += gg_b * gradient * std::log(T(1) - x) * pdf;

  T d_grad_b_da = log_sq_int_ab
                - log_int_b * (psi_a - psi_ab)
                - log_int_a * (psi_b - psi_ab)
                + I_x * (psi_a - psi_ab) * (psi_b - psi_ab)
                + I_x * psi1_ab;
  new_grad_a += gg_b * gradient * d_grad_b_da;

  T log_sq_int_bb = detail::log_squared_weighted_beta_integral(x, a, b, 1);
  T d_grad_b_db = log_sq_int_bb
                - T(2) * log_int_b * (psi_b - psi_ab)
                + I_x * (psi_b - psi_ab) * (psi_b - psi_ab)
                - I_x * (psi1_b - psi1_ab);
  new_grad_b += gg_b * gradient * d_grad_b_db;

  return {gg_out, new_grad_x, new_grad_a, new_grad_b};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
incomplete_beta_backward_backward(
    c10::complex<T> gg_x, c10::complex<T> gg_a, c10::complex<T> gg_b,
    c10::complex<T> gradient, c10::complex<T> x, c10::complex<T> a, c10::complex<T> b) {

  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));

  c10::complex<T> gg_out = zero;
  c10::complex<T> new_grad_x = zero;
  c10::complex<T> new_grad_a = zero;
  c10::complex<T> new_grad_b = zero;

  T eps = detail::incomplete_beta_eps<T>();
  if (std::abs(x) < eps || std::abs(x - one) < eps) {
    return {gg_out, new_grad_x, new_grad_a, new_grad_b};
  }

  c10::complex<T> log_beta_val = detail::log_beta(a, b);
  c10::complex<T> log_pdf = (a - one) * std::log(x) + (b - one) * std::log(one - x) - log_beta_val;
  c10::complex<T> pdf = std::exp(log_pdf);

  c10::complex<T> I_x = incomplete_beta(x, a, b);

  c10::complex<T> psi_a = digamma(a);
  c10::complex<T> psi_b = digamma(b);
  c10::complex<T> psi_ab = digamma(a + b);
  c10::complex<T> psi1_a = trigamma(a);
  c10::complex<T> psi1_b = trigamma(b);
  c10::complex<T> psi1_ab = trigamma(a + b);

  // gg_x contributions
  gg_out += gg_x * pdf;

  c10::complex<T> d_pdf_dx = pdf * ((a - one) / x - (b - one) / (one - x));
  new_grad_x += gg_x * gradient * d_pdf_dx;

  c10::complex<T> d_pdf_da = pdf * (std::log(x) - (psi_a - psi_ab));
  new_grad_a += gg_x * gradient * d_pdf_da;

  c10::complex<T> d_pdf_db = pdf * (std::log(one - x) - (psi_b - psi_ab));
  new_grad_b += gg_x * gradient * d_pdf_db;

  // gg_a contributions
  c10::complex<T> grad_a_over_g = -I_x * (psi_a - psi_ab);
  gg_out += gg_a * grad_a_over_g;

  new_grad_x += gg_a * gradient * std::log(x) * pdf;

  c10::complex<T> d_grad_a_da = I_x * (psi_a - psi_ab) * (psi_a - psi_ab) - I_x * (psi1_a - psi1_ab);
  new_grad_a += gg_a * gradient * d_grad_a_da;

  c10::complex<T> d_grad_a_db = I_x * (psi_a - psi_ab) * (psi_b - psi_ab) + I_x * psi1_ab;
  new_grad_b += gg_a * gradient * d_grad_a_db;

  // gg_b contributions
  c10::complex<T> grad_b_over_g = -I_x * (psi_b - psi_ab);
  gg_out += gg_b * grad_b_over_g;

  new_grad_x += gg_b * gradient * std::log(one - x) * pdf;

  c10::complex<T> d_grad_b_da = I_x * (psi_a - psi_ab) * (psi_b - psi_ab) + I_x * psi1_ab;
  new_grad_a += gg_b * gradient * d_grad_b_da;

  c10::complex<T> d_grad_b_db = I_x * (psi_b - psi_ab) * (psi_b - psi_ab) - I_x * (psi1_b - psi1_ab);
  new_grad_b += gg_b * gradient * d_grad_b_db;

  return {gg_out, new_grad_x, new_grad_a, new_grad_b};
}

} // namespace torchscience::kernel::special_functions
