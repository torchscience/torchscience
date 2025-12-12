#pragma once

#include "torchscience/csrc/cpu/macros.h"
#include "torchscience/csrc/impl/special_functions/log_beta.h"

TORCHSCIENCE_BINARY_CPU_KERNEL(log_beta, a, b)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(log_beta)
