#pragma once

#include "macros.h"

QUANTIZED_CPU_UNARY_OPERATOR(special_functions, gamma, z)

QUANTIZED_CPU_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)
