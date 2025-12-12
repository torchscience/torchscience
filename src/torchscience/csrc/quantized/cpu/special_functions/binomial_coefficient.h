#pragma once

#include <torchscience/csrc/impl/special_functions/binomial_coefficient.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(binomial_coefficient, n, k)

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(binomial_coefficient)

} // namespace torchscience::quantized::cpu::special_functions
