#pragma once

#include <torchscience/csrc/impl/special_functions/cos_pi.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t cos_pi(scalar_t x) {
  return torchscience::impl::special_functions::cos_pi(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(cos_pi)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(cos_pi)

} // namespace torchscience::quantized::cpu::special_functions
