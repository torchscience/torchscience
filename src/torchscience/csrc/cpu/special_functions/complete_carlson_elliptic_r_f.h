#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/complete_carlson_elliptic_r_f.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(complete_carlson_elliptic_r_f, x, y)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(complete_carlson_elliptic_r_f)

} // namespace torchscience::cpu::special_functions
