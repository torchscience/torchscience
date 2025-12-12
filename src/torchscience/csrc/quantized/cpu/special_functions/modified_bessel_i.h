#pragma once

#include "torchscience/csrc/impl/special_functions/modified_bessel_i.h"
#include "torchscience/csrc/quantized/cpu/macros.h"

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(modified_bessel_i, nu, x)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(modified_bessel_i)
