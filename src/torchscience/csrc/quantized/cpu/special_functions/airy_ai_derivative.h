#pragma once

#include <torchscience/csrc/impl/special_functions/airy_ai_derivative.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t airy_ai_derivative(scalar_t x) {
  return torchscience::impl::special_functions::airy_ai_derivative(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(airy_ai_derivative)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(airy_ai_derivative)

} // namespace torchscience::quantized::cpu::special_functions
