#pragma once

#include "torchscience/csrc/cpu/macros.h"
#include "torchscience/csrc/impl/special_functions/complete_elliptic_integral_pi.h"

TORCHSCIENCE_BINARY_CPU_KERNEL(complete_elliptic_integral_pi, n, k)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(complete_elliptic_integral_pi)
