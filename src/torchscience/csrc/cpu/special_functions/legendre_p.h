#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/legendre_p.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(legendre_p, n, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(legendre_p)

} // namespace torchscience::cpu::special_functions
