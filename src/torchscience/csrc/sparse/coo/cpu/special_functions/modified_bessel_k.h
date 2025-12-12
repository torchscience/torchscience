#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(modified_bessel_k, nu, x)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(modified_bessel_k)

} // namespace torchscience::sparse::coo::cpu::special_functions
