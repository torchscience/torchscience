#pragma once

#include "creation_operators.h"

#include "../cpu/noise.h"

TORCH_LIBRARY_IMPL(torchscience, Meta, m_meta_noise) {
    m_meta_noise.impl(
        "blue_noise",
        &::torchscience::meta::MetaStochasticCreationOperator<
            ::torchscience::cpu::BlueNoiseCPU>::forward<const at::Tensor&, int64_t>);
    m_meta_noise.impl(
        "brown_noise",
        &::torchscience::meta::MetaStochasticCreationOperator<
            ::torchscience::cpu::BrownNoiseCPU>::forward<const at::Tensor&, int64_t>);
    m_meta_noise.impl(
        "pink_noise",
        &::torchscience::meta::MetaStochasticCreationOperator<
            ::torchscience::cpu::PinkNoiseCPU>::forward<const at::Tensor&, int64_t>);
    m_meta_noise.impl(
        "violet_noise",
        &::torchscience::meta::MetaStochasticCreationOperator<
            ::torchscience::cpu::VioletNoiseCPU>::forward<const at::Tensor&, int64_t>);
    m_meta_noise.impl(
        "white_noise",
        &::torchscience::meta::MetaStochasticCreationOperator<
            ::torchscience::cpu::WhiteNoiseCPU>::forward<const at::Tensor&, int64_t>);
}
