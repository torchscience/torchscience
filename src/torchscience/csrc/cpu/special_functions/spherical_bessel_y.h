#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/spherical_bessel_y.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(spherical_bessel_y, n, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(spherical_bessel_y)

} // namespace torchscience::cpu::special_functions
