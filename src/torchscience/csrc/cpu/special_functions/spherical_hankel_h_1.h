#pragma once

#include "torchscience/csrc/cpu/macros.h"
#include "torchscience/csrc/impl/special_functions/spherical_hankel_h_1.h"

TORCHSCIENCE_BINARY_CPU_KERNEL(spherical_hankel_h_1, n, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(spherical_hankel_h_1)
