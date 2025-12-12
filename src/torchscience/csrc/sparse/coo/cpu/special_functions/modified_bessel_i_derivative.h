#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(modified_bessel_i_derivative, nu, x)

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(modified_bessel_i_derivative)

} // namespace torchscience::sparse::coo::cpu::special_functions
