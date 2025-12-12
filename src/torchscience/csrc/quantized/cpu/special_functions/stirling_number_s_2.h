#pragma once

#include <torchscience/csrc/impl/special_functions/stirling_number_s_2.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(stirling_number_s_2, n, k)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(stirling_number_s_2)

} // namespace torchscience::quantized::cpu::special_functions
