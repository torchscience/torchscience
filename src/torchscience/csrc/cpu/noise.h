#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <ATen/Generator.h>
#include <torch/library.h>

#include "creation_operators.h"
#include "../kernel/noise/pink_noise.h"

namespace torchscience::cpu {

struct PinkNoiseCPU {
    static std::vector<int64_t> output_shape(
        const at::Tensor& /*anchor*/,
        int64_t size
    ) {
        TORCH_CHECK(size > 0, "pink_noise: size must be positive, got ", size);
        return {size};
    }

    template<typename scalar_t>
    static void kernel(
        scalar_t* out,
        int64_t numel,
        c10::optional<at::Generator> generator,
        const at::Tensor& /*anchor*/,
        int64_t size
    ) {
        TORCH_INTERNAL_ASSERT(numel == size, "pink_noise: numel mismatch");
        auto opts = at::TensorOptions()
            .device(at::kCPU)
            .dtype(c10::CppTypeToScalarType<scalar_t>::value);
        at::Tensor out_tensor = at::from_blob(out, {size}, opts);
        torchscience::kernel::noise::pink_noise(out_tensor, generator);
    }
};

}  // namespace torchscience::cpu

TORCH_LIBRARY_IMPL(torchscience, CPU, m_cpu_noise) {
    m_cpu_noise.impl(
        "pink_noise",
        &::torchscience::cpu::CPUStochasticCreationOperator<
            ::torchscience::cpu::PinkNoiseCPU>::forward<const at::Tensor&, int64_t>);
}
