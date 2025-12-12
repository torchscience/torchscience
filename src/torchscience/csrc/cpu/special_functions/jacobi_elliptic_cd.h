#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/jacobi_elliptic_cd.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(jacobi_elliptic_cd, u, k)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(jacobi_elliptic_cd)

} // namespace torchscience::cpu::special_functions
