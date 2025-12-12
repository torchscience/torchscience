#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/polygamma.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(polygamma, n, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(polygamma)

} // namespace torchscience::cpu::special_functions
