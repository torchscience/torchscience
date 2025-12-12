#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(double_factorial)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(double_factorial)

} // namespace torchscience::sparse::coo::cpu::special_functions
