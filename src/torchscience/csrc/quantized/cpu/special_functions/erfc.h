#pragma once

#include <torchscience/csrc/impl/special_functions/erfc.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t erfc(scalar_t x) {
  return torchscience::impl::special_functions::erfc(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(erfc)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(erfc)

} // namespace torchscience::quantized::cpu::special_functions
