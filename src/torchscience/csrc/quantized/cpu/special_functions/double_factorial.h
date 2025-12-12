#pragma once

#include <torchscience/csrc/impl/special_functions/double_factorial.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(double_factorial)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(double_factorial)

} // namespace torchscience::quantized::cpu::special_functions
