#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/euler_totient_phi.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_CPU_KERNEL(euler_totient_phi)
TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(euler_totient_phi)

} // namespace torchscience::cpu::special_functions
