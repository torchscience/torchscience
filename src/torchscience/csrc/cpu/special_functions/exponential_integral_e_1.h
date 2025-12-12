#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/exponential_integral_e_1.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T exponential_integral_e_1(T x) {
  return torchscience::impl::special_functions::exponential_integral_e_1(x);
}

template <typename T>
T exponential_integral_e_1_backward(T x) {
  return torchscience::impl::special_functions::exponential_integral_e_1_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(exponential_integral_e_1)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(exponential_integral_e_1)

} // namespace torchscience::cpu::special_functions
