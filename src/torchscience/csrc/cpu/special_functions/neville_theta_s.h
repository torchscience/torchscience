#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/neville_theta_s.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(neville_theta_s, k, u)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(neville_theta_s)

} // namespace torchscience::cpu::special_functions
