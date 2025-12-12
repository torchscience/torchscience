#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/riemann_zeta.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_CPU_KERNEL(riemann_zeta)
TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(riemann_zeta)

} // namespace torchscience::cpu::special_functions
