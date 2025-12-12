#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/spherical_hankel_h_1.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(spherical_hankel_h_1, n, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(spherical_hankel_h_1)

} // namespace torchscience::cpu::special_functions
