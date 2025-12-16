#pragma once

#include "../impl/special_functions/chebyshev_polynomial_t.h"
#include "../impl/special_functions/gamma.h"

#include "macros.h"

CPU_UNARY_OPERATOR(special_functions, gamma, z)

CPU_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)
