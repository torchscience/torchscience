#pragma once

#include <torchscience/csrc/impl/special_functions/airy_ai.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t airy_ai(scalar_t x) {
  return torchscience::impl::special_functions::airy_ai(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(airy_ai)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(airy_ai)

} // namespace torchscience::quantized::cpu::special_functions
