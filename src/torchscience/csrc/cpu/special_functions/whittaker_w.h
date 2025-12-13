#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/whittaker_w.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_CPU_KERNEL(whittaker_w, kappa, mu, z)
TORCHSCIENCE_TERNARY_CPU_KERNEL_IMPL(whittaker_w)

} // namespace torchscience::cpu::special_functions
