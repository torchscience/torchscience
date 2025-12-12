#pragma once

#include <torchscience/csrc/impl/special_functions/airy_bi_derivative.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t airy_bi_derivative(scalar_t x) {
  return torchscience::impl::special_functions::airy_bi_derivative(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(airy_bi_derivative)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(airy_bi_derivative)

} // namespace torchscience::quantized::cpu::special_functions
