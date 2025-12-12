#pragma once

#include "torchscience/csrc/cpu/macros.h"
#include "torchscience/csrc/impl/special_functions/double_factorial.h"

TORCHSCIENCE_UNARY_CPU_KERNEL(double_factorial)
TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(double_factorial)
