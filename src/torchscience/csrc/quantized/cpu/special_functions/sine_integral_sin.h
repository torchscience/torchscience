#pragma once

#include <torchscience/csrc/impl/special_functions/sine_integral_sin.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t sine_integral_sin(scalar_t x) {
  return torchscience::impl::special_functions::sine_integral_sin(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(sine_integral_sin)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(sine_integral_sin)

} // namespace torchscience::quantized::cpu::special_functions
