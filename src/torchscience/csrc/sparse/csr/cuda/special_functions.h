#pragma once

#include "operators.cuh"

// Template-based registration (Sparse CSR CUDA operators delegate to dense ops)
TORCH_LIBRARY_IMPL(torchscience, SparseCsrCUDA, m_sparse_csr_cuda_special_functions) {
    REGISTER_SPARSE_CSR_CUDA_UNARY(m_sparse_csr_cuda_special_functions, gamma);
    REGISTER_SPARSE_CSR_CUDA_BINARY(m_sparse_csr_cuda_special_functions, chebyshev_polynomial_t);
    REGISTER_SPARSE_CSR_CUDA_TERNARY(m_sparse_csr_cuda_special_functions, incomplete_beta);
    REGISTER_SPARSE_CSR_CUDA_QUATERNARY(m_sparse_csr_cuda_special_functions, hypergeometric_2_f_1);
}
