#pragma once

#include "macros.h"

AUTOCAST_UNARY_OPERATOR(special_functions, gamma, z)

AUTOCAST_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)

AUTOCAST_TERNARY_OPERATOR(special_functions, incomplete_beta, z, a, b)

AUTOCAST_QUATERNARY_OPERATOR(special_functions, hypergeometric_2_f_1, a, b, c, z)
