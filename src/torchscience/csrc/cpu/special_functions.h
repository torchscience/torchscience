#pragma once

#include "../impl/special_functions/chebyshev_polynomial_t.h"
#include "../impl/special_functions/gamma.h"
#include "../impl/special_functions/incomplete_beta.h"

#include "macros.h"

CPU_UNARY_OPERATOR(special_functions, gamma, z)

CPU_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)

CPU_TERNARY_OPERATOR(special_functions, incomplete_beta, z, a, b)
