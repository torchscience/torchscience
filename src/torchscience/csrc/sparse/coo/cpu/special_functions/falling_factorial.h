#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(falling_factorial, x, n)

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(falling_factorial)

} // namespace torchscience::sparse::coo::cpu::special_functions
