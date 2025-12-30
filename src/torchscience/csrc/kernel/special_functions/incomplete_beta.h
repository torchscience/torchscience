#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "log_gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
constexpr T incomplete_beta_eps();

template <>
constexpr float incomplete_beta_eps<float>() { return 1e-7f; }

template <>
constexpr double incomplete_beta_eps<double>() { return 1e-15; }

template <>
inline c10::Half incomplete_beta_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 incomplete_beta_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
constexpr int incomplete_beta_max_iter() { return 200; }

template <typename T>
T log_beta(T a, T b) {
  return log_gamma(a) + log_gamma(b) - log_gamma(a + b);
}

template <typename T>
c10::complex<T> log_beta(c10::complex<T> a, c10::complex<T> b) {
  return log_gamma(a) + log_gamma(b) - log_gamma(a + b);
}

template <typename T>
T beta_continued_fraction(T a, T b, T x) {
  const T eps = incomplete_beta_eps<T>();
  const int max_iter = incomplete_beta_max_iter<T>();

  T qab = a + b;
  T qap = a + T(1);
  T qam = a - T(1);

  T c = T(1);
  T d = T(1) - qab * x / qap;
  if (std::abs(d) < eps) d = eps;
  d = T(1) / d;
  T h = d;

  for (int m = 1; m <= max_iter; ++m) {
    T m_t = static_cast<T>(m);
    T m2 = T(2) * m_t;

    T aa = m_t * (b - m_t) * x / ((qam + m2) * (a + m2));
    d = T(1) + aa * d;
    if (std::abs(d) < eps) d = eps;
    c = T(1) + aa / c;
    if (std::abs(c) < eps) c = eps;
    d = T(1) / d;
    h *= d * c;

    aa = -(a + m_t) * (qab + m_t) * x / ((a + m2) * (qap + m2));
    d = T(1) + aa * d;
    if (std::abs(d) < eps) d = eps;
    c = T(1) + aa / c;
    if (std::abs(c) < eps) c = eps;
    d = T(1) / d;
    T delta = d * c;
    h *= delta;

    if (std::abs(delta - T(1)) < eps) break;
  }

  return h;
}

template <typename T>
c10::complex<T> beta_continued_fraction(c10::complex<T> a, c10::complex<T> b, c10::complex<T> x) {
  const T eps = incomplete_beta_eps<T>();
  const int max_iter = incomplete_beta_max_iter<T>();

  c10::complex<T> qab = a + b;
  c10::complex<T> qap = a + c10::complex<T>(T(1), T(0));
  c10::complex<T> qam = a - c10::complex<T>(T(1), T(0));

  c10::complex<T> one(T(1), T(0));
  c10::complex<T> c = one;
  c10::complex<T> d = one - qab * x / qap;
  if (std::abs(d) < eps) d = c10::complex<T>(eps, T(0));
  d = one / d;
  c10::complex<T> h = d;

  for (int m = 1; m <= max_iter; ++m) {
    c10::complex<T> m_c(static_cast<T>(m), T(0));
    c10::complex<T> m2 = c10::complex<T>(T(2), T(0)) * m_c;

    c10::complex<T> aa = m_c * (b - m_c) * x / ((qam + m2) * (a + m2));
    d = one + aa * d;
    if (std::abs(d) < eps) d = c10::complex<T>(eps, T(0));
    c = one + aa / c;
    if (std::abs(c) < eps) c = c10::complex<T>(eps, T(0));
    d = one / d;
    h *= d * c;

    aa = -(a + m_c) * (qab + m_c) * x / ((a + m2) * (qap + m2));
    d = one + aa * d;
    if (std::abs(d) < eps) d = c10::complex<T>(eps, T(0));
    c = one + aa / c;
    if (std::abs(c) < eps) c = c10::complex<T>(eps, T(0));
    d = one / d;
    c10::complex<T> delta = d * c;
    h *= delta;

    if (std::abs(delta - one) < eps) break;
  }

  return h;
}

template <typename T>
T incomplete_beta_cf(T a, T b, T x) {
  T log_prefix = a * std::log(x) + b * std::log(T(1) - x) - log_beta(a, b);
  T prefix = std::exp(log_prefix);
  T cf = beta_continued_fraction(a, b, x);
  return prefix * cf / a;
}

template <typename T>
c10::complex<T> incomplete_beta_cf(c10::complex<T> a, c10::complex<T> b, c10::complex<T> x) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> log_prefix = a * std::log(x) + b * std::log(one - x) - log_beta(a, b);
  c10::complex<T> prefix = std::exp(log_prefix);
  c10::complex<T> cf = beta_continued_fraction(a, b, x);
  return prefix * cf / a;
}

} // namespace detail

template <typename T>
T incomplete_beta(T x, T a, T b) {
  if (x <= T(0)) {
    return T(0);
  }

  if (x >= T(1)) {
    return T(1);
  }

  if (a == T(1) && b == T(1)) {
    return x;
  }

  if (a == T(1)) {
    return T(1) - std::pow(T(1) - x, b);
  }

  if (b == T(1)) {
    return std::pow(x, a);
  }

  T threshold = (a + T(1)) / (a + b + T(2));

  if (x > threshold) {
    return T(1) - incomplete_beta(T(1) - x, b, a);
  }

  return detail::incomplete_beta_cf(a, b, x);
}

template <typename T>
c10::complex<T> incomplete_beta(c10::complex<T> x, c10::complex<T> a, c10::complex<T> b) {
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));

  if (std::abs(x) < detail::incomplete_beta_eps<T>()) return zero;

  if (std::abs(x - one) < detail::incomplete_beta_eps<T>()) return one;

  if (std::abs(a - one) < detail::incomplete_beta_eps<T>() &&
      std::abs(b - one) < detail::incomplete_beta_eps<T>()) {
    return x;
  }

  if (std::abs(a - one) < detail::incomplete_beta_eps<T>()) {
    return one - std::pow(one - x, b);
  }

  if (std::abs(b - one) < detail::incomplete_beta_eps<T>()) {
    return std::pow(x, a);
  }

  T threshold = (a.real() + T(1)) / (a.real() + b.real() + T(2));

  if (x.real() > threshold && std::abs(x.imag()) < detail::incomplete_beta_eps<T>()) {
    return one - incomplete_beta(one - x, b, a);
  }

  return detail::incomplete_beta_cf(a, b, x);
}

} // namespace torchscience::kernel::special_functions
