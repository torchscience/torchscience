#pragma once

#include <torchscience/csrc/impl/special_functions/tangent_number_t_2.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(tangent_number_t_2)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(tangent_number_t_2)

} // namespace torchscience::quantized::cpu::special_functions
