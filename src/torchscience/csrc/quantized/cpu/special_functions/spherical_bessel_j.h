#pragma once

#include "torchscience/csrc/impl/special_functions/spherical_bessel_j.h"
#include "torchscience/csrc/quantized/cpu/macros.h"

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(spherical_bessel_j, n, x)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(spherical_bessel_j)
