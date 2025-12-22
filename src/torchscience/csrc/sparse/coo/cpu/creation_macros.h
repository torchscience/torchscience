#pragma once

#include <ATen/ATen.h>
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
// SPARSE_COO_CPU_CREATION_OPERATOR
// =============================================================================
// Flexible macro for creating sparse COO tensors from scalar parameters.
// Used for sparse structured matrices (identity, diagonal, etc.)
//
// Parameters:
//   NAMESPACE     - Namespace (e.g., structured_matrix)
//   OPERATOR_NAME - Operator name (e.g., sparse_identity)
//   OUTPUT_SHAPE  - Shape expression (e.g., {n, n})
//   PARAMS        - Parenthesized typed params: (int64_t n) or ()
//   ARGS          - Parenthesized arg names: (n) or ()
//
// Kernel signature expected:
//   namespace impl::NAMESPACE {
//     template <typename scalar_t>
//     std::pair<at::Tensor, at::Tensor> OPERATOR_NAME_kernel(
//       int64_t* shape, int64_t ndim, PARAMS...
//     )
//     // Returns (indices, values) tensors
//   }
// =============================================================================

#define SPARSE_COO_CPU_CREATION_OPERATOR(                                       \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  OUTPUT_SHAPE,                                                                 \
  PARAMS,                                                                       \
  ARGS                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::sparse::coo::cpu::NAMESPACE {                           \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  TORCHSCIENCE_UNPACK(PARAMS)                                                   \
  TORCHSCIENCE_COMMA_IF(PARAMS)                                                 \
  const c10::optional<at::ScalarType>& dtype,                                   \
  const c10::optional<at::Layout>& layout,                                      \
  const c10::optional<at::Device>& device,                                      \
  bool requires_grad                                                            \
) {                                                                             \
  (void)layout;  /* Sparse layout is implicit */                                \
                                                                                \
  auto options = at::TensorOptions()                                            \
    .dtype(dtype.value_or(                                                      \
      c10::typeMetaToScalarType(at::get_default_dtype())                        \
    ))                                                                          \
    .device(device.value_or(at::kCPU))                                          \
    .requires_grad(false);                                                      \
                                                                                \
  std::vector<int64_t> shape_vec = OUTPUT_SHAPE;                                \
  for (auto s : shape_vec) {                                                    \
    TORCH_CHECK(s >= 0,                                                         \
      #OPERATOR_NAME ": size must be non-negative, got ", s);                   \
  }                                                                             \
                                                                                \
  at::Tensor indices;                                                           \
  at::Tensor values;                                                            \
                                                                                \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                              \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    options.dtype().toScalarType(),                                             \
    #OPERATOR_NAME,                                                             \
    [&]() {                                                                     \
      auto result = impl::NAMESPACE::OPERATOR_NAME##_kernel<scalar_t>(          \
        shape_vec.data(),                                                       \
        static_cast<int64_t>(shape_vec.size())                                  \
        TORCHSCIENCE_COMMA_IF(ARGS)                                             \
        TORCHSCIENCE_UNPACK(ARGS)                                               \
      );                                                                        \
      indices = result.first;                                                   \
      values = result.second.to(options);                                       \
    }                                                                           \
  );                                                                            \
                                                                                \
  at::Tensor output = at::_sparse_coo_tensor_unsafe(                            \
    indices,                                                                    \
    values,                                                                     \
    shape_vec,                                                                  \
    options.layout(at::kSparse)                                                 \
  );                                                                            \
                                                                                \
  if (requires_grad) {                                                          \
    output = output.requires_grad_(true);                                       \
  }                                                                             \
                                                                                \
  return output;                                                                \
}                                                                               \
                                                                                \
}  /* namespace torchscience::sparse::coo::cpu::NAMESPACE */                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, SparseCPU, module) {                           \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::sparse::coo::cpu::NAMESPACE::OPERATOR_NAME                   \
  );                                                                            \
}
