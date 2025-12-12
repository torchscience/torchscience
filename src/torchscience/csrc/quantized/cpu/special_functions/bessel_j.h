#pragma once

#include "torchscience/csrc/impl/special_functions/bessel_j.h"
#include "torchscience/csrc/quantized/cpu/macros.h"

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(bessel_j, nu, x)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(bessel_j)
