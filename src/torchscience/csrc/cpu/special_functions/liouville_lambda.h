#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/liouville_lambda.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_CPU_KERNEL(liouville_lambda)
TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(liouville_lambda)

} // namespace torchscience::cpu::special_functions
