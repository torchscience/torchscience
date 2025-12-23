#pragma once

#include "operators.h"

// Template-based registration (Sparse CSR CPU operators delegate to dense ops)
TORCH_LIBRARY_IMPL(torchscience, SparseCsrCPU, m_sparse_csr_cpu_special_functions) {
    REGISTER_SPARSE_CSR_CPU_UNARY(m_sparse_csr_cpu_special_functions, gamma);
    REGISTER_SPARSE_CSR_CPU_BINARY(m_sparse_csr_cpu_special_functions, chebyshev_polynomial_t);
    REGISTER_SPARSE_CSR_CPU_TERNARY(m_sparse_csr_cpu_special_functions, incomplete_beta);
    REGISTER_SPARSE_CSR_CPU_QUATERNARY(m_sparse_csr_cpu_special_functions, hypergeometric_2_f_1);
}
