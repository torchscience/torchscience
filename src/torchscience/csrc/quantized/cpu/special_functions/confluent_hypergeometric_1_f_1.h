#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL(confluent_hypergeometric_1_f_1, a, b, z)
TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL_IMPL(confluent_hypergeometric_1_f_1)

} // namespace torchscience::quantized::cpu::special_functions
