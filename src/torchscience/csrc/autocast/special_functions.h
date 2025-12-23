#pragma once

#include "operators.h"

// Template-based registration (Autocast operators don't need ImplTraits - dtype casting only)
TORCH_LIBRARY_IMPL(torchscience, Autocast, m_autocast_special_functions) {
    REGISTER_AUTOCAST_UNARY(m_autocast_special_functions, gamma);
    REGISTER_AUTOCAST_BINARY(m_autocast_special_functions, chebyshev_polynomial_t);
    REGISTER_AUTOCAST_TERNARY(m_autocast_special_functions, incomplete_beta);
    REGISTER_AUTOCAST_QUATERNARY(m_autocast_special_functions, hypergeometric_2_f_1);
}
