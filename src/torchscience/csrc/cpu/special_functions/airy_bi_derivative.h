#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/airy_bi_derivative.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T airy_bi_derivative(T x) {
  return torchscience::impl::special_functions::airy_bi_derivative(x);
}

template <typename T>
T airy_bi_derivative_backward(T x) {
  return torchscience::impl::special_functions::airy_bi_derivative_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(airy_bi_derivative)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(airy_bi_derivative)

} // namespace torchscience::cpu::special_functions
