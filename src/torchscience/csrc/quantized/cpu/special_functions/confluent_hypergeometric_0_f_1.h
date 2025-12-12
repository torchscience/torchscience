#pragma once

#include <torchscience/csrc/impl/special_functions/confluent_hypergeometric_0_f_1.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(confluent_hypergeometric_0_f_1, b, z)

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(confluent_hypergeometric_0_f_1)
