#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/kelvin_bei.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(kelvin_bei, v, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(kelvin_bei)

} // namespace torchscience::cpu::special_functions
