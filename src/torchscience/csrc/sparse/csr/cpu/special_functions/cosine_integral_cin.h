#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(cosine_integral_cin)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(cosine_integral_cin)

} // namespace torchscience::sparse::csr::cpu::special_functions
