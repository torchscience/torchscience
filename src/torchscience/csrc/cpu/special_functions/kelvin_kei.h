#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/kelvin_kei.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(kelvin_kei, v, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(kelvin_kei)

} // namespace torchscience::cpu::special_functions
