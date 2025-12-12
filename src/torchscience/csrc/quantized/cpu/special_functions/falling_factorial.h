#pragma once

#include <torchscience/csrc/impl/special_functions/falling_factorial.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(falling_factorial, x, n)

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(falling_factorial)

} // namespace torchscience::quantized::cpu::special_functions
