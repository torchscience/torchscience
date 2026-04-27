#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <ATen/Generator.h>
#include <torch/library.h>

#include "creation_operators.h"
#include "../kernel/noise/blue_noise.h"
#include "../kernel/noise/brown_noise.h"
#include "../kernel/noise/grey_noise.h"
#include "../kernel/noise/pink_noise.h"
#include "../kernel/noise/violet_noise.h"
#include "../kernel/noise/white_noise.h"

namespace torchscience::cpu {

// Common shape contract for all single-argument 1-D colored-noise ops.
#define DEFINE_NOISE_CPU_TRAITS(NAME, KERNEL_FN)                                   \
    struct NAME {                                                                  \
        static std::vector<int64_t> output_shape(                                  \
            const at::Tensor& /*anchor*/,                                          \
            int64_t size                                                           \
        ) {                                                                        \
            TORCH_CHECK(size > 0, #KERNEL_FN ": size must be positive, got ",      \
                        size);                                                     \
            return {size};                                                         \
        }                                                                          \
                                                                                   \
        template<typename scalar_t>                                                \
        static void kernel(                                                        \
            scalar_t* out,                                                         \
            int64_t numel,                                                         \
            c10::optional<at::Generator> generator,                                \
            const at::Tensor& /*anchor*/,                                          \
            int64_t size                                                           \
        ) {                                                                        \
            TORCH_INTERNAL_ASSERT(numel == size,                                   \
                                  #KERNEL_FN ": numel mismatch");                  \
            auto opts = at::TensorOptions()                                        \
                .device(at::kCPU)                                                  \
                .dtype(c10::CppTypeToScalarType<scalar_t>::value);                 \
            at::Tensor out_tensor = at::from_blob(out, {size}, opts);              \
            torchscience::kernel::noise::KERNEL_FN(out_tensor, generator);         \
        }                                                                          \
    }

DEFINE_NOISE_CPU_TRAITS(BlueNoiseCPU,   blue_noise);
DEFINE_NOISE_CPU_TRAITS(BrownNoiseCPU,  brown_noise);
DEFINE_NOISE_CPU_TRAITS(PinkNoiseCPU,   pink_noise);
DEFINE_NOISE_CPU_TRAITS(VioletNoiseCPU, violet_noise);
DEFINE_NOISE_CPU_TRAITS(WhiteNoiseCPU,  white_noise);

#undef DEFINE_NOISE_CPU_TRAITS

// Grey noise has an extra ``sample_rate`` parameter (A-weighting is defined
// in absolute Hz), so it does not fit the macro above and is defined
// explicitly here.
struct GreyNoiseCPU {
    static std::vector<int64_t> output_shape(
        const at::Tensor& /*anchor*/,
        int64_t size,
        double /*sample_rate*/
    ) {
        TORCH_CHECK(size > 0,
                    "grey_noise: size must be positive, got ", size);
        return {size};
    }

    template<typename scalar_t>
    static void kernel(
        scalar_t* out,
        int64_t numel,
        c10::optional<at::Generator> generator,
        const at::Tensor& /*anchor*/,
        int64_t size,
        double sample_rate
    ) {
        TORCH_INTERNAL_ASSERT(numel == size, "grey_noise: numel mismatch");
        auto opts = at::TensorOptions()
            .device(at::kCPU)
            .dtype(c10::CppTypeToScalarType<scalar_t>::value);
        at::Tensor out_tensor = at::from_blob(out, {size}, opts);
        torchscience::kernel::noise::grey_noise(out_tensor, generator, sample_rate);
    }
};

}  // namespace torchscience::cpu

TORCH_LIBRARY_IMPL(torchscience, CPU, m_cpu_noise) {
    m_cpu_noise.impl(
        "blue_noise",
        &::torchscience::cpu::CPUStochasticCreationOperator<
            ::torchscience::cpu::BlueNoiseCPU>::forward<const at::Tensor&, int64_t>);
    m_cpu_noise.impl(
        "brown_noise",
        &::torchscience::cpu::CPUStochasticCreationOperator<
            ::torchscience::cpu::BrownNoiseCPU>::forward<const at::Tensor&, int64_t>);
    m_cpu_noise.impl(
        "grey_noise",
        &::torchscience::cpu::CPUStochasticCreationOperator<
            ::torchscience::cpu::GreyNoiseCPU>::forward<const at::Tensor&, int64_t, double>);
    m_cpu_noise.impl(
        "pink_noise",
        &::torchscience::cpu::CPUStochasticCreationOperator<
            ::torchscience::cpu::PinkNoiseCPU>::forward<const at::Tensor&, int64_t>);
    m_cpu_noise.impl(
        "violet_noise",
        &::torchscience::cpu::CPUStochasticCreationOperator<
            ::torchscience::cpu::VioletNoiseCPU>::forward<const at::Tensor&, int64_t>);
    m_cpu_noise.impl(
        "white_noise",
        &::torchscience::cpu::CPUStochasticCreationOperator<
            ::torchscience::cpu::WhiteNoiseCPU>::forward<const at::Tensor&, int64_t>);
}
