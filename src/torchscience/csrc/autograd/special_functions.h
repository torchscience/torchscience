#pragma once

#include "macros.h"

AUTOGRAD_UNARY_OPERATOR(special_functions, Gamma, gamma, z)

AUTOGRAD_BINARY_OPERATOR(special_functions, ChebyshevPolynomialT, chebyshev_polynomial_t, v, z)

AUTOGRAD_TERNARY_OPERATOR(special_functions, IncompleteBeta, incomplete_beta, z, a, b)
