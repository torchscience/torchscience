#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(sin_pi)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(sin_pi)

} // namespace torchscience::sparse::csr::cpu::special_functions
