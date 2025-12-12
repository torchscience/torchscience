#pragma once

#include <torchscience/csrc/impl/special_functions/sinc_pi.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t sinc_pi(scalar_t x) {
  return torchscience::impl::special_functions::sinc_pi(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(sinc_pi)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(sinc_pi)

} // namespace torchscience::quantized::cpu::special_functions
