#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../../core/creation_common.h"

namespace torchscience::sparse::coo::cpu {

// SparseCOOCreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t>
//     static std::pair<at::Tensor, at::Tensor> kernel(int64_t* shape, int64_t ndim, params...);
//     Returns (indices, values) tensors

template<typename CreationTraits>
struct SparseCOOCPUCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        (void)layout;  // Sparse layout is implicit

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "sparse_coo_cpu_creation_op");

        auto options = at::TensorOptions()
            .dtype(dtype.value_or(
                c10::typeMetaToScalarType(at::get_default_dtype())
            ))
            .device(device.value_or(at::kCPU))
            .requires_grad(false);

        at::Tensor indices;
        at::Tensor values;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16,
            at::kHalf,
            options.dtype().toScalarType(),
            "sparse_coo_cpu_creation",
            [&]() {
                auto result = CreationTraits::template kernel<scalar_t>(
                    shape_vec.data(),
                    static_cast<int64_t>(shape_vec.size()),
                    args...
                );
                indices = result.first;
                values = result.second.to(options);
            }
        );

        at::Tensor output = at::_sparse_coo_tensor_unsafe(
            indices,
            values,
            shape_vec,
            options.layout(at::kSparse)
        );

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_SPARSE_COO_CPU_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::sparse::coo::cpu::SparseCOOCPUCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::sparse::coo::cpu
