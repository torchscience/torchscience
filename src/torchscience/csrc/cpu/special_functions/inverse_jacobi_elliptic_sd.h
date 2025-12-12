#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/inverse_jacobi_elliptic_sd.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(inverse_jacobi_elliptic_sd, x, k)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(inverse_jacobi_elliptic_sd)

} // namespace torchscience::cpu::special_functions
