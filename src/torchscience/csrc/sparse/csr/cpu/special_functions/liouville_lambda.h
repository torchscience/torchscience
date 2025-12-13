#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/liouville_lambda.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(liouville_lambda)
TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(liouville_lambda)

} // namespace torchscience::sparse::csr::cpu::special_functions
