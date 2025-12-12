#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/hyperbolic_sine_integral_shi.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T hyperbolic_sine_integral_shi(T x) {
  return torchscience::impl::special_functions::hyperbolic_sine_integral_shi(x);
}

template <typename T>
T hyperbolic_sine_integral_shi_backward(T x) {
  return torchscience::impl::special_functions::hyperbolic_sine_integral_shi_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(hyperbolic_sine_integral_shi)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(hyperbolic_sine_integral_shi)

} // namespace torchscience::cpu::special_functions
