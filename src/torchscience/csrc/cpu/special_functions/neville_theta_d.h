#pragma once

#include "torchscience/csrc/cpu/macros.h"
#include "torchscience/csrc/impl/special_functions/neville_theta_d.h"

TORCHSCIENCE_BINARY_CPU_KERNEL(neville_theta_d, k, u)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(neville_theta_d)
