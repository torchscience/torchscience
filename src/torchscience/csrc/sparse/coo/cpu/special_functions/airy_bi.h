#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(airy_bi)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(airy_bi)

} // namespace torchscience::sparse::coo::cpu::special_functions
