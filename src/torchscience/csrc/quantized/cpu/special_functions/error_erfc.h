#pragma once

#include <torchscience/csrc/impl/special_functions/error_erfc.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t error_erfc(scalar_t x) {
  return torchscience::impl::special_functions::error_erfc(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(error_erfc)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(error_erfc)

} // namespace torchscience::quantized::cpu::special_functions
