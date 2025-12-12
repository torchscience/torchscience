#pragma once

#include "torchscience/csrc/cpu/macros.h"
#include "torchscience/csrc/impl/special_functions/binomial_coefficient.h"

TORCHSCIENCE_BINARY_CPU_KERNEL(binomial_coefficient, n, k)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(binomial_coefficient)
