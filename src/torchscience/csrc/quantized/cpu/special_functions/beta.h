#pragma once

#include "torchscience/csrc/impl/special_functions/beta.h"
#include "torchscience/csrc/quantized/cpu/macros.h"

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(beta, a, b)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(beta)
