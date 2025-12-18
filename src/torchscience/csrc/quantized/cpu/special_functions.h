#pragma once

#include "macros.h"

QUANTIZED_CPU_UNARY_OPERATOR(special_functions, gamma, z)

QUANTIZED_CPU_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)

QUANTIZED_CPU_TERNARY_OPERATOR(special_functions, incomplete_beta, z, a, b)

QUANTIZED_CPU_QUATERNARY_OPERATOR(special_functions, hypergeometric_2_f_1, a, b, c, z)
