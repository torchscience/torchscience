#pragma once

#include <torchscience/csrc/impl/special_functions/legendre_q.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(legendre_q, n, x)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(legendre_q)

} // namespace torchscience::quantized::cpu::special_functions
