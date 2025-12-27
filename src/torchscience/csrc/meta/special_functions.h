#pragma once

#include "../core/pointwise_registration.h"
#include "../operators/special_functions.def"

TORCH_LIBRARY_IMPL(torchscience, Meta, m_meta_special_functions) {
    #define REGISTER_OP(name, arity, impl) REGISTER_POINTWISE_META(m_meta_special_functions, name, arity);
    TORCHSCIENCE_SPECIAL_FUNCTIONS(REGISTER_OP)
    #undef REGISTER_OP
}
