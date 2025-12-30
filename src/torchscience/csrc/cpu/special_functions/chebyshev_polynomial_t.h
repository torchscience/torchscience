#pragma once

#include "macros.h"

#include "../../kernel/special_functions/chebyshev_polynomial_t.h"
#include "../../kernel/special_functions/chebyshev_polynomial_t_backward.h"
#include "../../kernel/special_functions/chebyshev_polynomial_t_backward_backward.h"

TORCHSCIENCE_CPU_BINARY_OPERATOR(chebyshev_polynomial_t, x, n)
