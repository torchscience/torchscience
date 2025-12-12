#pragma once

#include <torchscience/csrc/impl/special_functions/jacobi_elliptic_cd.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(jacobi_elliptic_cd, u, k)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(jacobi_elliptic_cd)

} // namespace torchscience::quantized::cpu::special_functions
