#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/spherical_modified_bessel_k.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(spherical_modified_bessel_k, n, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(spherical_modified_bessel_k)

} // namespace torchscience::cpu::special_functions
