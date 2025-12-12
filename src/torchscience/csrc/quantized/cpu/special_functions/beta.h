#pragma once

#include <torchscience/csrc/impl/special_functions/beta.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(beta, a, b)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(beta)

} // namespace torchscience::quantized::cpu::special_functions
