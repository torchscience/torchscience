#pragma once

#include "operators.h"

// Template-based registration (Meta operators don't need ImplTraits - shape only)
TORCH_LIBRARY_IMPL(torchscience, Meta, m_meta_special_functions) {
    REGISTER_META_UNARY(m_meta_special_functions, gamma);
    REGISTER_META_BINARY(m_meta_special_functions, chebyshev_polynomial_t);
    REGISTER_META_TERNARY(m_meta_special_functions, incomplete_beta);
    REGISTER_META_QUATERNARY(m_meta_special_functions, hypergeometric_2_f_1);
}
