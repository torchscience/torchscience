#pragma once

#include "torchscience/csrc/cpu/macros.h"
#include "torchscience/csrc/impl/special_functions/confluent_hypergeometric_0_f_1.h"

TORCHSCIENCE_BINARY_CPU_KERNEL(confluent_hypergeometric_0_f_1, b, z)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(confluent_hypergeometric_0_f_1)
