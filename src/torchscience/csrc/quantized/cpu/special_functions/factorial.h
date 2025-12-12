#pragma once

#include <torchscience/csrc/impl/special_functions/factorial.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(factorial)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(factorial)

} // namespace torchscience::quantized::cpu::special_functions
