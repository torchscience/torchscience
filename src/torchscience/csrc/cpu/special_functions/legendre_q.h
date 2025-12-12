#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/legendre_q.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(legendre_q, n, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(legendre_q)

} // namespace torchscience::cpu::special_functions
