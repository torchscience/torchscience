#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/kelvin_ker.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(kelvin_ker, v, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(kelvin_ker)

} // namespace torchscience::cpu::special_functions
