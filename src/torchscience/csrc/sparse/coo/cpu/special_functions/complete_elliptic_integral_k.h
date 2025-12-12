#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(complete_elliptic_integral_k)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(complete_elliptic_integral_k)

} // namespace torchscience::sparse::coo::cpu::special_functions
