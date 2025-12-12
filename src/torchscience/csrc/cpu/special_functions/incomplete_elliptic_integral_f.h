#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/incomplete_elliptic_integral_f.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(incomplete_elliptic_integral_f, phi, k)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(incomplete_elliptic_integral_f)

} // namespace torchscience::cpu::special_functions
