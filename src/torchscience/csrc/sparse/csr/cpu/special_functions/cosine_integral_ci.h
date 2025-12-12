#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(cosine_integral_ci)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(cosine_integral_ci)

} // namespace torchscience::sparse::csr::cpu::special_functions
