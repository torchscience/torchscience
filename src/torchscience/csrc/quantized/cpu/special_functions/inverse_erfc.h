#pragma once

#include <torchscience/csrc/impl/special_functions/inverse_erfc.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t inverse_erfc(scalar_t x) {
  return torchscience::impl::special_functions::inverse_erfc(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(inverse_erfc)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(inverse_erfc)

} // namespace torchscience::quantized::cpu::special_functions
