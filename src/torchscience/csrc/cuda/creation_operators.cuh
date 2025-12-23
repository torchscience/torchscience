#pragma once

#include <vector>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "../core/creation_common.h"

namespace torchscience::cuda {

// CreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t> static void launch_kernel(scalar_t* out, int64_t numel, params...);

template<typename CreationTraits>
struct CUDACreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        auto dev = device.value_or(at::kCUDA);
        TORCH_CHECK(dev.is_cuda(), "device must be CUDA");

        c10::cuda::CUDAGuard guard(dev);

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "cuda_creation_op");

        auto options = ::torchscience::core::build_options(dtype, layout, device, dev);
        int64_t numel = ::torchscience::core::compute_numel(shape_vec);

        at::Tensor output = at::empty(shape_vec, options);

        if (numel > 0) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16,
                at::kHalf,
                output.scalar_type(),
                "cuda_creation",
                [&]() {
                    CreationTraits::template launch_kernel<scalar_t>(
                        output.data_ptr<scalar_t>(),
                        numel,
                        args...
                    );
                }
            );
        }

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_CUDA_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::cuda::CUDACreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::cuda
