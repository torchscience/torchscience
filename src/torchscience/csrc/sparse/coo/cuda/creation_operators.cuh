#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "../../../core/creation_common.h"

namespace torchscience::sparse::coo::cuda {

template<typename CreationTraits>
struct SparseCOOCUDACreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        (void)layout;

        auto dev = device.value_or(at::kCUDA);
        TORCH_CHECK(dev.is_cuda(), "device must be CUDA");

        c10::cuda::CUDAGuard guard(dev);

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "sparse_coo_cuda_creation_op");

        auto options = at::TensorOptions()
            .dtype(dtype.value_or(
                c10::typeMetaToScalarType(at::get_default_dtype())
            ))
            .device(dev)
            .requires_grad(false);

        at::Tensor indices;
        at::Tensor values;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16,
            at::kHalf,
            options.dtype().toScalarType(),
            "sparse_coo_cuda_creation",
            [&]() {
                auto result = CreationTraits::template kernel<scalar_t>(
                    shape_vec.data(),
                    static_cast<int64_t>(shape_vec.size()),
                    args...
                );
                indices = result.first.to(dev);
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

#define REGISTER_SPARSE_COO_CUDA_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::sparse::coo::cuda::SparseCOOCUDACreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::sparse::coo::cuda
