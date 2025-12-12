#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(neville_theta_s, k, u)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(neville_theta_s)

} // namespace torchscience::quantized::cpu::special_functions
