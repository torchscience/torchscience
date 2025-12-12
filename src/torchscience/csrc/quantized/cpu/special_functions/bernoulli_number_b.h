#pragma once

#include <torchscience/csrc/impl/special_functions/bernoulli_number_b.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(bernoulli_number_b)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(bernoulli_number_b)

} // namespace torchscience::quantized::cpu::special_functions
