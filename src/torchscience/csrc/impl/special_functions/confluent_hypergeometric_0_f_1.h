#pragma once

#include <boost/math/special_functions/hypergeometric_0F1.hpp>

template <typename T>
T confluent_hypergeometric_0_f_1(T b, T z) {
  // Confluent hypergeometric limit function 0F1(; b; z)
  return boost::math::hypergeometric_0F1(b, z);
}

template <typename T>
std::tuple<T, T> confluent_hypergeometric_0_f_1_backward(T b, T z) {
  // Gradient with respect to b is complex, set to zero for now
  T grad_b = T(0);

  // d/dz 0F1(; b; z) = 0F1(; b+1; z) / b
  T f1 = boost::math::hypergeometric_0F1(b + T(1), z);
  T grad_z = f1 / b;

  return std::make_tuple(grad_b, grad_z);
}
