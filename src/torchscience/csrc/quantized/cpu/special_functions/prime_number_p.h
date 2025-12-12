#pragma once

#include <torchscience/csrc/impl/special_functions/prime_number_p.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(prime_number_p)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(prime_number_p)

} // namespace torchscience::quantized::cpu::special_functions
