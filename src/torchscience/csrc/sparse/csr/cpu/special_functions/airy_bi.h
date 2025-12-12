#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(airy_bi)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(airy_bi)

} // namespace torchscience::sparse::csr::cpu::special_functions
