#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/complete_elliptic_integral_pi.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(complete_elliptic_integral_pi, n, k)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(complete_elliptic_integral_pi)

} // namespace torchscience::cpu::special_functions
