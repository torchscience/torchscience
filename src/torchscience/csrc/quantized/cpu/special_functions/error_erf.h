#pragma once

#include <torchscience/csrc/impl/special_functions/error_erf.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t error_erf(scalar_t x) {
  return torchscience::impl::special_functions::error_erf(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(error_erf)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(error_erf)

} // namespace torchscience::quantized::cpu::special_functions
