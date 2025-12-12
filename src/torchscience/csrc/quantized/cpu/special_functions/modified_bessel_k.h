#pragma once

#include <torchscience/csrc/impl/special_functions/modified_bessel_k.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(modified_bessel_k, nu, x)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(modified_bessel_k)

} // namespace torchscience::quantized::cpu::special_functions
