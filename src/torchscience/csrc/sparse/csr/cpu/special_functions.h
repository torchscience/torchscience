#pragma once

#include "macros.h"

SPARSE_CSR_CPU_UNARY_OPERATOR(special_functions, gamma, z)

SPARSE_CSR_CPU_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)

SPARSE_CSR_CPU_TERNARY_OPERATOR(special_functions, incomplete_beta, z, a, b)
