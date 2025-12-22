#pragma once

#include <ATen/Dispatch.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>
#include "../core/creation_common.h"

// Helper macros for parenthesized parameter lists
#define TORCHSCIENCE_UNPACK_IMPL(...) __VA_ARGS__
#define TORCHSCIENCE_UNPACK(X) TORCHSCIENCE_UNPACK_IMPL X
#define TORCHSCIENCE_COMMA_IF_IMPL(...) __VA_OPT__(,)
#define TORCHSCIENCE_COMMA_IF(X) TORCHSCIENCE_COMMA_IF_IMPL(TORCHSCIENCE_UNPACK(X))

#define CPU_CREATION_OPERATOR(                                                  \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  OUTPUT_SHAPE,                                                                 \
  PARAMS,                                                                       \
  ARGS                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::cpu::NAMESPACE {                                        \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  TORCHSCIENCE_UNPACK(PARAMS)                                                   \
  TORCHSCIENCE_COMMA_IF(PARAMS)                                                 \
  const c10::optional<at::ScalarType>& dtype,                                   \
  const c10::optional<at::Layout>& layout,                                      \
  const c10::optional<at::Device>& device,                                      \
  bool requires_grad                                                            \
) {                                                                             \
  std::vector<int64_t> shape_vec = OUTPUT_SHAPE;                                \
  ::torchscience::core::check_size_nonnegative(shape_vec, #OPERATOR_NAME);      \
                                                                                \
  auto options = ::torchscience::core::build_options(                           \
    dtype, layout, device, at::kCPU                                             \
  );                                                                            \
                                                                                \
  int64_t numel = ::torchscience::core::compute_numel(shape_vec);               \
                                                                                \
  at::Tensor output = at::empty(shape_vec, options);                            \
                                                                                \
  if (numel > 0) {                                                              \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                            \
      at::kBFloat16,                                                            \
      at::kHalf,                                                                \
      output.scalar_type(),                                                     \
      #OPERATOR_NAME,                                                           \
      [&]() {                                                                   \
        impl::NAMESPACE::OPERATOR_NAME##_kernel<scalar_t>(                      \
          output.data_ptr<scalar_t>(),                                          \
          numel                                                                 \
          TORCHSCIENCE_COMMA_IF(ARGS)                                           \
          TORCHSCIENCE_UNPACK(ARGS)                                             \
        );                                                                      \
      }                                                                         \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (requires_grad) {                                                          \
    output = output.requires_grad_(true);                                       \
  }                                                                             \
                                                                                \
  return output;                                                                \
}                                                                               \
                                                                                \
}  /* namespace torchscience::cpu::NAMESPACE */                                 \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                                 \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::cpu::NAMESPACE::OPERATOR_NAME                                \
  );                                                                            \
}
