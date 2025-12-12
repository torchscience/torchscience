#pragma once

#include <torchscience/csrc/impl/special_functions/log_beta.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t log_beta(scalar_t a, scalar_t b) {
  return ::log_beta(a, b);
}

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(log_beta, a, b)

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(log_beta)

} // namespace torchscience::quantized::cpu::special_functions
