#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/sine_integral_si.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T sine_integral_si(T x) {
  return torchscience::impl::special_functions::sine_integral_si(x);
}

template <typename T>
T sine_integral_si_backward(T x) {
  return torchscience::impl::special_functions::sine_integral_si_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(sine_integral_si)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(sine_integral_si)

} // namespace torchscience::cpu::special_functions
