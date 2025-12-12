#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/carlson_elliptic_r_c.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(carlson_elliptic_r_c, x, y)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(carlson_elliptic_r_c)

} // namespace torchscience::cpu::special_functions
