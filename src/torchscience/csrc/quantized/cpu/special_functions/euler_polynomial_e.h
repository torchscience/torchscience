#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/euler_polynomial_e.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(euler_polynomial_e)

} // namespace torchscience::quantized::cpu::special_functions
