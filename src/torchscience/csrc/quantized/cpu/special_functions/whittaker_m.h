#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL(whittaker_m, kappa, mu, z)
TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL_IMPL(whittaker_m)

} // namespace torchscience::quantized::cpu::special_functions
