#pragma once

#include <torchscience/csrc/impl/special_functions/fibonacci_number_f.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(fibonacci_number_f)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(fibonacci_number_f)

} // namespace torchscience::quantized::cpu::special_functions
