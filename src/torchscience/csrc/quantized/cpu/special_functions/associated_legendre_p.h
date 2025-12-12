#pragma once

#include <torchscience/csrc/impl/special_functions/associated_legendre_p.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL(associated_legendre_p, n, m, x)
TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL_IMPL(associated_legendre_p)

} // namespace torchscience::quantized::cpu::special_functions
