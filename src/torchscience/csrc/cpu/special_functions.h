#pragma once

#include "../core/pointwise_registration.h"
#include "../operators/special_functions.def"

// Include all impl traits
#include "../impl/special_functions/gamma_traits.h"
#include "../impl/special_functions/chebyshev_polynomial_t_traits.h"
#include "../impl/special_functions/incomplete_beta_traits.h"
#include "../impl/special_functions/hypergeometric_2_f_1_traits.h"

TORCH_LIBRARY_IMPL(torchscience, CPU, m_cpu_special_functions) {
    #define REGISTER_OP(name, arity, impl) REGISTER_POINTWISE_CPU(m_cpu_special_functions, name, arity, impl);
    TORCHSCIENCE_SPECIAL_FUNCTIONS(REGISTER_OP)
    #undef REGISTER_OP
}
