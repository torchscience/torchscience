#pragma once

#include "torchscience/csrc/impl/special_functions/spherical_modified_bessel_k.h"
#include "torchscience/csrc/quantized/cpu/macros.h"

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(spherical_modified_bessel_k, n, x)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(spherical_modified_bessel_k)
