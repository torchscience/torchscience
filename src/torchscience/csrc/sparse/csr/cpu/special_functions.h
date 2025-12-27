#pragma once

#include "operators.h"
#include "../../../operators/special_functions.def"

// Arity-based dispatch helper for Sparse CSR CPU
#define REGISTER_SPARSE_CSR_CPU_ARITY_1(m, name) REGISTER_SPARSE_CSR_CPU_UNARY(m, name)
#define REGISTER_SPARSE_CSR_CPU_ARITY_2(m, name) REGISTER_SPARSE_CSR_CPU_BINARY(m, name)
#define REGISTER_SPARSE_CSR_CPU_ARITY_3(m, name) REGISTER_SPARSE_CSR_CPU_TERNARY(m, name)
#define REGISTER_SPARSE_CSR_CPU_ARITY_4(m, name) REGISTER_SPARSE_CSR_CPU_QUATERNARY(m, name)

#define REGISTER_SPARSE_CSR_CPU_DISPATCH(m, name, arity) \
    REGISTER_SPARSE_CSR_CPU_ARITY_##arity(m, name)

TORCH_LIBRARY_IMPL(torchscience, SparseCsrCPU, m_sparse_csr_cpu_special_functions) {
    #define REGISTER_OP(name, arity, impl) REGISTER_SPARSE_CSR_CPU_DISPATCH(m_sparse_csr_cpu_special_functions, name, arity);
    TORCHSCIENCE_SPECIAL_FUNCTIONS(REGISTER_OP)
    #undef REGISTER_OP
}
