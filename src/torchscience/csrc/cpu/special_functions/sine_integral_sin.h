#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/sine_integral_sin.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T sine_integral_sin(T x) {
  return torchscience::impl::special_functions::sine_integral_sin(x);
}

template <typename T>
T sine_integral_sin_backward(T x) {
  return torchscience::impl::special_functions::sine_integral_sin_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(sine_integral_sin)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(sine_integral_sin)

} // namespace torchscience::cpu::special_functions
