#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <tuple>
#include <array>

#include "digamma.h"
#include "incomplete_beta.h"
#include "log_gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
constexpr T gauss_kronrod_eps();

template <>
constexpr float gauss_kronrod_eps<float>() { return 1e-6f; }

template <>
constexpr double gauss_kronrod_eps<double>() { return 1e-12; }

template <>
inline c10::Half gauss_kronrod_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 gauss_kronrod_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
struct GaussKronrodNodes15 {
  static constexpr int n_gauss = 7;
  static constexpr int n_kronrod = 15;

  static constexpr double kronrod_nodes[15] = {
    -0.991455371120812639206854697526329,
    -0.949107912342758524526189684047851,
    -0.864864423359769072789712788640926,
    -0.741531185599394439863864773280788,
    -0.586087235467691130294144838258730,
    -0.405845151377397166906606412076961,
    -0.207784955007898467600689403773245,
     0.0,
     0.207784955007898467600689403773245,
     0.405845151377397166906606412076961,
     0.586087235467691130294144838258730,
     0.741531185599394439863864773280788,
     0.864864423359769072789712788640926,
     0.949107912342758524526189684047851,
     0.991455371120812639206854697526329
  };

  static constexpr double kronrod_weights[15] = {
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714,
    0.204432940075298892414161999234649,
    0.190350578064785409913256402421014,
    0.169004726639267902826583426598550,
    0.140653259715525918745189590510238,
    0.104790010322250183839876322541518,
    0.063092092629978553290700663189204,
    0.022935322010529224963732008058970
  };

  static constexpr double gauss_weights[7] = {
    0.129484966168869693270611432679082,
    0.279705391489276667901467771423780,
    0.381830050505118944950369775488975,
    0.417959183673469387755102040816327,
    0.381830050505118944950369775488975,
    0.279705391489276667901467771423780,
    0.129484966168869693270611432679082
  };

  static constexpr int gauss_indices[7] = {1, 3, 5, 7, 9, 11, 13};
};

template <typename T, typename F>
std::pair<T, T> gauss_kronrod_15(F f, T lower, T upper) {
  T center = (upper + lower) / T(2);
  T half_length = (upper - lower) / T(2);

  T gauss_sum = T(0);
  T kronrod_sum = T(0);

  for (int i = 0; i < 15; ++i) {
    T node = center + half_length * static_cast<T>(GaussKronrodNodes15<T>::kronrod_nodes[i]);
    T f_val = f(node);
    kronrod_sum += static_cast<T>(GaussKronrodNodes15<T>::kronrod_weights[i]) * f_val;
  }

  for (int i = 0; i < 7; ++i) {
    int idx = GaussKronrodNodes15<T>::gauss_indices[i];
    T node = center + half_length * static_cast<T>(GaussKronrodNodes15<T>::kronrod_nodes[idx]);
    T f_val = f(node);
    gauss_sum += static_cast<T>(GaussKronrodNodes15<T>::gauss_weights[i]) * f_val;
  }

  T result = half_length * kronrod_sum;
  T error = std::abs(half_length * (kronrod_sum - gauss_sum));

  return {result, error};
}

template <typename T, typename F>
T adaptive_integrate(F f, T lower, T upper, T tol, int max_depth) {
  auto [result, error] = gauss_kronrod_15<T>(f, lower, upper);

  if (error < tol || max_depth <= 0) {
    return result;
  }

  T mid = (lower + upper) / T(2);
  T left = adaptive_integrate(f, lower, mid, tol / T(2), max_depth - 1);
  T right = adaptive_integrate(f, mid, upper, tol / T(2), max_depth - 1);

  return left + right;
}

template <typename T>
T log_weighted_beta_integral(T x, T a, T b, bool weight_log_t) {
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

  auto integrand = [a, b, weight_log_t](T t) -> T {
    if (t <= T(0) || t >= T(1)) return T(0);

    T log_integrand = (a - T(1)) * std::log(t) + (b - T(1)) * std::log(T(1) - t);
    T base_val = std::exp(log_integrand);

    if (weight_log_t) {
      return std::log(t) * base_val;
    } else {
      return std::log(T(1) - t) * base_val;
    }
  };

  T integral = adaptive_integrate(integrand, lower, upper, tol, max_depth);

  return integral / beta_val;
}

} // namespace detail

template <typename T>
std::tuple<T, T, T> incomplete_beta_backward(T gradient, T x, T a, T b) {
  if (x <= T(0) || x >= T(1)) {
    return {T(0), T(0), T(0)};
  }

  T log_beta_val = detail::log_beta(a, b);
  T beta_val = std::exp(log_beta_val);

  T log_pdf = (a - T(1)) * std::log(x) + (b - T(1)) * std::log(T(1) - x) - log_beta_val;
  T pdf = std::exp(log_pdf);
  T grad_x = gradient * pdf;

  T I_x = incomplete_beta(x, a, b);

  T psi_a = digamma(a);
  T psi_b = digamma(b);
  T psi_ab = digamma(a + b);

  T log_integral_a = detail::log_weighted_beta_integral(x, a, b, true);
  T grad_a = gradient * (log_integral_a - I_x * (psi_a - psi_ab));

  T log_integral_b = detail::log_weighted_beta_integral(x, a, b, false);
  T grad_b = gradient * (log_integral_b - I_x * (psi_b - psi_ab));

  return {
    grad_x,
    grad_a,
    grad_b
  };
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> incomplete_beta_backward(
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> a,
    c10::complex<T> b
) {
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));

  T eps = detail::incomplete_beta_eps<T>();
  if (std::abs(x) < eps || std::abs(x - one) < eps) {
    return {zero, zero, zero};
  }

  c10::complex<T> log_beta_val = detail::log_beta(a, b);

  c10::complex<T> log_pdf = (a - one) * std::log(x) + (b - one) * std::log(one - x) - log_beta_val;
  c10::complex<T> pdf = std::exp(log_pdf);

  c10::complex<T> I_x = incomplete_beta(x, a, b);

  c10::complex<T> psi_ab = digamma(a + b);

  return {
    gradient * pdf,
    -gradient * I_x * (digamma(a) - psi_ab),
    -gradient * I_x * (digamma(b) - psi_ab)
  };
}

} // namespace torchscience::kernel::special_functions
