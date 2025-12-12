#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/bessel_y_derivative.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(bessel_y_derivative, nu, x)

TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(bessel_y_derivative)

} // namespace torchscience::cpu::special_functions
