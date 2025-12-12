#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(beta, a, b)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(beta)

} // namespace torchscience::sparse::coo::cpu::special_functions
