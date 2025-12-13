#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/bernoulli_polynomial_b.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(bernoulli_polynomial_b)

} // namespace torchscience::quantized::cpu::special_functions
