#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/exponential_integral_ei.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T exponential_integral_ei(T x) {
  return torchscience::impl::special_functions::exponential_integral_ei(x);
}

template <typename T>
T exponential_integral_ei_backward(T x) {
  return torchscience::impl::special_functions::exponential_integral_ei_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(exponential_integral_ei)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(exponential_integral_ei)

} // namespace torchscience::cpu::special_functions
