#pragma once

#include "torchscience/csrc/cpu/macros.h"
#include "torchscience/csrc/impl/special_functions/incomplete_legendre_elliptic_integral_d.h"

TORCHSCIENCE_BINARY_CPU_KERNEL(incomplete_legendre_elliptic_integral_d, phi, k)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(incomplete_legendre_elliptic_integral_d)
