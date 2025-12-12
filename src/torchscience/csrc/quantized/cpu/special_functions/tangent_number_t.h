#pragma once

#include <torchscience/csrc/impl/special_functions/tangent_number_t.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(tangent_number_t)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(tangent_number_t)

} // namespace torchscience::quantized::cpu::special_functions
