#pragma once

#include <torchscience/csrc/impl/special_functions/sine_integral_si.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t sine_integral_si(scalar_t x) {
  return torchscience::impl::special_functions::sine_integral_si(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(sine_integral_si)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(sine_integral_si)

} // namespace torchscience::quantized::cpu::special_functions
