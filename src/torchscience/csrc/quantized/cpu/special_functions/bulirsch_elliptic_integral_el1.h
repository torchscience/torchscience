#pragma once

#include <torchscience/csrc/impl/special_functions/bulirsch_elliptic_integral_el1.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(bulirsch_elliptic_integral_el1, x, kc)

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(bulirsch_elliptic_integral_el1)
