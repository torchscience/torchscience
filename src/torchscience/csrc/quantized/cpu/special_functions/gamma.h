#pragma once

#include <torchscience/csrc/impl/special_functions/gamma.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t gamma(scalar_t x) {
  return torchscience::impl::special_functions::gamma(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(gamma)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(gamma)

} // namespace torchscience::quantized::cpu::special_functions
