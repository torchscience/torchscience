#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(exponential_integral_e, n, x)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(exponential_integral_e)

} // namespace torchscience::sparse::coo::cpu::special_functions
