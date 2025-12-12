#pragma once

#include <torchscience/csrc/impl/special_functions/rising_factorial.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(rising_factorial, x, n)

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(rising_factorial)

} // namespace torchscience::quantized::cpu::special_functions
