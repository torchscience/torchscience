#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/stirling_number_s_2.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(stirling_number_s_2, n, k)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(stirling_number_s_2)

} // namespace torchscience::cpu::special_functions
