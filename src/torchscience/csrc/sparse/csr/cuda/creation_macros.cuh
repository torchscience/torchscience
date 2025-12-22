#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// =============================================================================
// Helper macros for handling parenthesized parameter lists
// =============================================================================

#ifndef TORCHSCIENCE_UNPACK_IMPL
#define TORCHSCIENCE_UNPACK_IMPL(...) __VA_ARGS__
#define TORCHSCIENCE_UNPACK(X) TORCHSCIENCE_UNPACK_IMPL X
#define TORCHSCIENCE_COMMA_IF_IMPL(...) __VA_OPT__(, __VA_ARGS__)
#define TORCHSCIENCE_COMMA_IF(X) TORCHSCIENCE_COMMA_IF_IMPL(TORCHSCIENCE_UNPACK(X))
#endif

// =============================================================================
// SPARSE_CSR_CUDA_CREATION_OPERATOR
// =============================================================================
// Flexible macro for creating sparse CSR tensors on CUDA.
//
// Kernel signature expected:
//   namespace impl::NAMESPACE {
//     template <typename scalar_t>
//     std::tuple<at::Tensor, at::Tensor, at::Tensor> OPERATOR_NAME_kernel(
//       int64_t* shape, int64_t ndim, PARAMS...
//     )
//   }
// =============================================================================

#define SPARSE_CSR_CUDA_CREATION_OPERATOR(                                      \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  OUTPUT_SHAPE,                                                                 \
  PARAMS,                                                                       \
  ARGS                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::sparse::csr::cuda::NAMESPACE {                          \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  TORCHSCIENCE_UNPACK(PARAMS)                                                   \
  TORCHSCIENCE_COMMA_IF(PARAMS)                                                 \
  const c10::optional<at::ScalarType>& dtype,                                   \
  const c10::optional<at::Layout>& layout,                                      \
  const c10::optional<at::Device>& device,                                      \
  bool requires_grad                                                            \
) {                                                                             \
  (void)layout;                                                                 \
                                                                                \
  auto dev = device.value_or(at::kCUDA);                                        \
  TORCH_CHECK(dev.is_cuda(), #OPERATOR_NAME ": device must be CUDA");           \
                                                                                \
  c10::cuda::CUDAGuard guard(dev);                                              \
                                                                                \
  auto options = at::TensorOptions()                                            \
    .dtype(dtype.value_or(                                                      \
      c10::typeMetaToScalarType(at::get_default_dtype())                        \
    ))                                                                          \
    .device(dev)                                                                \
    .requires_grad(false);                                                      \
                                                                                \
  std::vector<int64_t> shape_vec = OUTPUT_SHAPE;                                \
  for (auto s : shape_vec) {                                                    \
    TORCH_CHECK(s >= 0,                                                         \
      #OPERATOR_NAME ": size must be non-negative, got ", s);                   \
  }                                                                             \
                                                                                \
  at::Tensor crow_indices;                                                      \
  at::Tensor col_indices;                                                       \
  at::Tensor values;                                                            \
                                                                                \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                              \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    options.dtype().toScalarType(),                                             \
    #OPERATOR_NAME "_cuda",                                                     \
    [&]() {                                                                     \
      auto result = impl::NAMESPACE::OPERATOR_NAME##_kernel<scalar_t>(          \
        shape_vec.data(),                                                       \
        static_cast<int64_t>(shape_vec.size())                                  \
        TORCHSCIENCE_COMMA_IF(ARGS)                                             \
        TORCHSCIENCE_UNPACK(ARGS)                                               \
      );                                                                        \
      crow_indices = std::get<0>(result).to(dev);                               \
      col_indices = std::get<1>(result).to(dev);                                \
      values = std::get<2>(result).to(options);                                 \
    }                                                                           \
  );                                                                            \
                                                                                \
  at::Tensor output = at::sparse_csr_tensor(                                    \
    crow_indices,                                                               \
    col_indices,                                                                \
    values,                                                                     \
    shape_vec,                                                                  \
    options                                                                     \
  );                                                                            \
                                                                                \
  if (requires_grad) {                                                          \
    output = output.requires_grad_(true);                                       \
  }                                                                             \
                                                                                \
  return output;                                                                \
}                                                                               \
                                                                                \
}  /* namespace torchscience::sparse::csr::cuda::NAMESPACE */                   \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, SparseCsrCUDA, module) {                       \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::sparse::csr::cuda::NAMESPACE::OPERATOR_NAME                  \
  );                                                                            \
}
