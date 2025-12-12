#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/complete_carlson_elliptic_r_g.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(complete_carlson_elliptic_r_g, x, y)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(complete_carlson_elliptic_r_g)

} // namespace torchscience::cpu::special_functions
