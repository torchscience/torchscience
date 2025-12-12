#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/modified_bessel_k_derivative.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(modified_bessel_k_derivative, nu, x)

TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(modified_bessel_k_derivative)

} // namespace torchscience::cpu::special_functions
