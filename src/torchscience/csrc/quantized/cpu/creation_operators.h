#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../core/creation_common.h"

namespace torchscience::quantized::cpu {

// QuantizedCreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t> static void kernel(scalar_t* out, int64_t numel, params...);

template<typename CreationTraits>
struct QuantizedCPUCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        double scale,
        int64_t zero_point,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        // Determine quantized type from requested dtype
        auto base_dtype = dtype.value_or(at::kFloat);
        at::ScalarType qtype;
        if (base_dtype == at::kFloat || base_dtype == at::kQInt8) {
            qtype = at::kQInt8;
        } else if (base_dtype == at::kQUInt8) {
            qtype = at::kQUInt8;
        } else if (base_dtype == at::kQInt32) {
            qtype = at::kQInt32;
        } else {
            qtype = at::kQInt8;
        }

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "quantized_cpu_creation_op");
        int64_t numel = ::torchscience::core::compute_numel(shape_vec);

        auto float_options = at::TensorOptions()
            .dtype(at::kFloat)
            .layout(layout.value_or(at::kStrided))
            .device(device.value_or(at::kCPU));

        // Create float tensor first, then quantize
        at::Tensor float_output = at::empty(shape_vec, float_options);

        if (numel > 0) {
            CreationTraits::template kernel<float>(
                float_output.data_ptr<float>(),
                numel,
                args...
            );
        }

        at::Tensor output = at::quantize_per_tensor(
            float_output, scale, zero_point, qtype
        );

        if (requires_grad) {
            TORCH_WARN("requires_grad ignored for quantized tensor");
        }

        return output;
    }
};

#define REGISTER_QUANTIZED_CPU_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::quantized::cpu::QuantizedCPUCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::quantized::cpu
