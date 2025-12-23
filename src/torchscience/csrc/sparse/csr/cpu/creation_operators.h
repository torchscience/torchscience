#pragma once

#include <vector>
#include <tuple>
#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../../core/creation_common.h"

namespace torchscience::sparse::csr::cpu {

// SparseCSRCreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t>
//     static std::tuple<at::Tensor, at::Tensor, at::Tensor> kernel(int64_t* shape, int64_t ndim, params...);
//     Returns (crow_indices, col_indices, values) tensors

template<typename CreationTraits>
struct SparseCSRCPUCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        (void)layout;

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "sparse_csr_cpu_creation_op");

        auto options = at::TensorOptions()
            .dtype(dtype.value_or(
                c10::typeMetaToScalarType(at::get_default_dtype())
            ))
            .device(device.value_or(at::kCPU))
            .requires_grad(false);

        at::Tensor crow_indices;
        at::Tensor col_indices;
        at::Tensor values;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16,
            at::kHalf,
            options.dtype().toScalarType(),
            "sparse_csr_cpu_creation",
            [&]() {
                auto result = CreationTraits::template kernel<scalar_t>(
                    shape_vec.data(),
                    static_cast<int64_t>(shape_vec.size()),
                    args...
                );
                crow_indices = std::get<0>(result);
                col_indices = std::get<1>(result);
                values = std::get<2>(result).to(options);
            }
        );

        at::Tensor output = at::sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            shape_vec,
            options
        );

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_SPARSE_CSR_CPU_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::sparse::csr::cpu::SparseCSRCPUCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::sparse::csr::cpu
