#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/associated_legendre_p.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_CPU_KERNEL(associated_legendre_p, n, m, x)
TORCHSCIENCE_TERNARY_CPU_KERNEL_IMPL(associated_legendre_p)

} // namespace torchscience::cpu::special_functions
