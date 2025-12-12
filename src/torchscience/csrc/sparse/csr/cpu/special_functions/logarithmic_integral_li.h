#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(logarithmic_integral_li)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(logarithmic_integral_li)

} // namespace torchscience::sparse::csr::cpu::special_functions
