#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_SPARSE_CSR_CPU_KERNEL(whittaker_w, kappa, mu, z)
TORCHSCIENCE_TERNARY_SPARSE_CSR_CPU_KERNEL_IMPL(whittaker_w)

} // namespace torchscience::sparse::csr::cpu::special_functions
