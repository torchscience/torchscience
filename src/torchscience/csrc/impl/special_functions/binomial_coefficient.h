#pragma once

#include <boost/math/special_functions/binomial.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

template <typename T>
T binomial_coefficient(T n, T k) {
  return boost::math::binomial_coefficient<T>(
    static_cast<unsigned int>(n),
    static_cast<unsigned int>(k)
  );
}

template <typename T>
std::tuple<T, T> binomial_coefficient_backward(T n, T k) {
  // Gradients for discrete arguments are not well-defined
  // Return zero gradients
  return std::make_tuple(T(0), T(0));
}

} // namespace torchscience::impl::special_functions
