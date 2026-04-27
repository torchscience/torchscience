#pragma once

#include "creation_operators.h"

#include "../cpu/noise.h"

TORCH_LIBRARY_IMPL(torchscience, Meta, m_meta_noise) {
    m_meta_noise.impl(
        "pink_noise",
        &::torchscience::meta::MetaStochasticCreationOperator<
            ::torchscience::cpu::PinkNoiseCPU>::forward<const at::Tensor&, int64_t>);
}
