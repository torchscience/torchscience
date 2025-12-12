#pragma once

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

namespace torchscience::impl::special_functions {

template <typename T>
T log_beta(T a, T b) {
  // log(B(a,b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a+b))
  return boost::math::lgamma(a) + boost::math::lgamma(b) - boost::math::lgamma(a + b);
}

template <typename T>
std::tuple<T, T> log_beta_backward(T a, T b) {
  // d/da log(B(a,b)) = psi(a) - psi(a+b)
  // d/db log(B(a,b)) = psi(b) - psi(a+b)
  T digamma_a = boost::math::digamma(a);
  T digamma_b = boost::math::digamma(b);
  T digamma_ab = boost::math::digamma(a + b);

  return std::make_tuple(
    digamma_a - digamma_ab,
    digamma_b - digamma_ab
  );
}

} // namespace torchscience::impl::special_functions
