#pragma once

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T beta(T a, T b) {
  return boost::math::beta(a, b);
}

template <typename T>
std::tuple<T, T> beta_backward(T a, T b) {
  T beta_val = boost::math::beta(a, b);
  T digamma_a = boost::math::digamma(a);
  T digamma_b = boost::math::digamma(b);
  T digamma_ab = boost::math::digamma(a + b);

  return std::make_tuple(
    beta_val * (digamma_a - digamma_ab),
    beta_val * (digamma_b - digamma_ab)
  );
}

} // namespace torchscience::impl::special_functions
