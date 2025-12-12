#pragma once

#include <torchscience/csrc/sparse/csr/cuda/macros.h>

namespace torchscience::sparse::csr::cuda::special_functions {

TORCHSCIENCE_UNARY_SPARSE_CSR_CUDA_KERNEL(cos_pi)

TORCHSCIENCE_UNARY_SPARSE_CSR_CUDA_KERNEL_IMPL(cos_pi)

} // namespace torchscience::sparse::csr::cuda::special_functions
