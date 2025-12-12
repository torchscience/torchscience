#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/bulirsch_elliptic_integral_el1.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(bulirsch_elliptic_integral_el1, x, kc)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(bulirsch_elliptic_integral_el1)

} // namespace torchscience::cpu::special_functions
