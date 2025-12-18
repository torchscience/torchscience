#pragma once

#include "macros.h"

META_UNARY_OPERATOR(special_functions, gamma, z)

META_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)

META_TERNARY_OPERATOR(special_functions, incomplete_beta, z, a, b)

META_QUATERNARY_OPERATOR(special_functions, hypergeometric_2_f_1, a, b, c, z)
