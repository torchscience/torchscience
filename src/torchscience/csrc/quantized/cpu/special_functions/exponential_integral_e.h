#pragma once

#include "torchscience/csrc/impl/special_functions/exponential_integral_e.h"
#include "torchscience/csrc/quantized/cpu/macros.h"

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(exponential_integral_e, n, x)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(exponential_integral_e)
