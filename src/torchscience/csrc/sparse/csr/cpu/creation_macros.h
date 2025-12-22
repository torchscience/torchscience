#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../../core/creation_common.h"

#ifndef TORCHSCIENCE_UNPACK_IMPL
#define TORCHSCIENCE_UNPACK_IMPL(...) __VA_ARGS__
#define TORCHSCIENCE_UNPACK(X) TORCHSCIENCE_UNPACK_IMPL X
#define TORCHSCIENCE_COMMA_IF_IMPL(...) __VA_OPT__(,)
#define TORCHSCIENCE_COMMA_IF(X) TORCHSCIENCE_COMMA_IF_IMPL(TORCHSCIENCE_UNPACK(X))
#endif

// Sparse CSR: kernel returns (crow_indices, col_indices, values) tuple
// Like PyTorch's empty_sparse_compressed
#define SPARSE_CSR_CPU_CREATION_OPERATOR(                                       \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  OUTPUT_SHAPE,                                                                 \
  PARAMS,                                                                       \
  ARGS                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::sparse::csr::cpu::NAMESPACE {                           \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  TORCHSCIENCE_UNPACK(PARAMS)                                                   \
  TORCHSCIENCE_COMMA_IF(PARAMS)                                                 \
  const c10::optional<at::ScalarType>& dtype,                                   \
  const c10::optional<at::Layout>& layout,                                      \
  const c10::optional<at::Device>& device,                                      \
  bool requires_grad                                                            \
) {                                                                             \
  (void)layout;  /* CSR layout is implicit */                                   \
                                                                                \
  std::vector<int64_t> shape_vec = OUTPUT_SHAPE;                                \
  ::torchscience::core::check_size_nonnegative(shape_vec, #OPERATOR_NAME);      \
                                                                                \
  auto options = ::torchscience::core::build_options(                           \
    dtype, c10::nullopt, device, at::kCPU                                       \
  );                                                                            \
                                                                                \
  at::Tensor crow_indices;                                                      \
  at::Tensor col_indices;                                                       \
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
      crow_indices = std::get<0>(result);                                       \
      col_indices = std::get<1>(result);                                        \
      values = std::get<2>(result).to(options);                                 \
    }                                                                           \
  );                                                                            \
                                                                                \
  /* Construct sparse CSR tensor like PyTorch's empty_sparse_compressed */      \
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
}  /* namespace torchscience::sparse::csr::cpu::NAMESPACE */                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, SparseCsrCPU, module) {                        \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::sparse::csr::cpu::NAMESPACE::OPERATOR_NAME                   \
  );                                                                            \
}
