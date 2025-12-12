#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/jacobi_amplitude_am.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(jacobi_amplitude_am, u, k)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(jacobi_amplitude_am)

} // namespace torchscience::cpu::special_functions
