#pragma once

#include "../core/pointwise_registration.h"
#include "../operators/special_functions.def"

TORCH_LIBRARY_IMPL(torchscience, Autocast, m_autocast_special_functions) {
    #define REGISTER_OP(name, arity, impl) REGISTER_POINTWISE_AUTOCAST(m_autocast_special_functions, name, arity);
    TORCHSCIENCE_SPECIAL_FUNCTIONS(REGISTER_OP)
    #undef REGISTER_OP
}
