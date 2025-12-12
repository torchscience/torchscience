#pragma once

#include <torchscience/csrc/impl/special_functions/log_gamma.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t log_gamma(scalar_t x) {
  return torchscience::impl::special_functions::log_gamma(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(log_gamma)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(log_gamma)

} // namespace torchscience::quantized::cpu::special_functions
