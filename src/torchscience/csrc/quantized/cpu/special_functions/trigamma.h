#pragma once

#include <torchscience/csrc/impl/special_functions/trigamma.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t trigamma(scalar_t x) {
  return torchscience::impl::special_functions::trigamma(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(trigamma)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(trigamma)

} // namespace torchscience::quantized::cpu::special_functions
