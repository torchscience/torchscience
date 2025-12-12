#pragma once

#include <torchscience/csrc/impl/special_functions/digamma.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t digamma(scalar_t x) {
  return torchscience::impl::special_functions::digamma(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(digamma)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(digamma)

} // namespace torchscience::quantized::cpu::special_functions
