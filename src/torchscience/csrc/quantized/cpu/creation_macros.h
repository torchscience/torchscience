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
// QUANTIZED_CPU_CREATION_OPERATOR
// =============================================================================
// Flexible macro for creating quantized tensors from scalar parameters.
//
// Kernel signature expected:
//   namespace impl::NAMESPACE {
//     template <typename scalar_t>
//     void OPERATOR_NAME_kernel(scalar_t* output, int64_t numel, PARAMS...)
//   }
// =============================================================================

#define QUANTIZED_CPU_CREATION_OPERATOR(                                        \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  OUTPUT_SHAPE,                                                                 \
  PARAMS,                                                                       \
  ARGS                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::quantized::cpu::NAMESPACE {                             \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  TORCHSCIENCE_UNPACK(PARAMS)                                                   \
  TORCHSCIENCE_COMMA_IF(PARAMS)                                                 \
  double scale,                                                                 \
  int64_t zero_point,                                                           \
  const c10::optional<at::ScalarType>& dtype,                                   \
  const c10::optional<at::Layout>& layout,                                      \
  const c10::optional<at::Device>& device,                                      \
  bool requires_grad                                                            \
) {                                                                             \
  auto base_dtype = dtype.value_or(at::kFloat);                                 \
  at::ScalarType qtype;                                                         \
  if (base_dtype == at::kFloat || base_dtype == at::kQInt8) {                   \
    qtype = at::kQInt8;                                                         \
  } else if (base_dtype == at::kQUInt8) {                                       \
    qtype = at::kQUInt8;                                                        \
  } else if (base_dtype == at::kQInt32) {                                       \
    qtype = at::kQInt32;                                                        \
  } else {                                                                      \
    qtype = at::kQInt8;                                                         \
  }                                                                             \
                                                                                \
  auto options = at::TensorOptions()                                            \
    .dtype(qtype)                                                               \
    .layout(layout.value_or(at::kStrided))                                      \
    .device(device.value_or(at::kCPU))                                          \
    .requires_grad(false);                                                      \
                                                                                \
  std::vector<int64_t> shape_vec = OUTPUT_SHAPE;                                \
  for (auto s : shape_vec) {                                                    \
    TORCH_CHECK(s >= 0,                                                         \
      #OPERATOR_NAME ": size must be non-negative, got ", s);                   \
  }                                                                             \
                                                                                \
  int64_t numel = 1;                                                            \
  for (auto s : shape_vec) {                                                    \
    numel *= s;                                                                 \
  }                                                                             \
                                                                                \
  /* Create float tensor first, then quantize */                                \
  at::Tensor float_output = at::empty(shape_vec,                                \
    options.dtype(at::kFloat));                                                 \
                                                                                \
  if (numel > 0) {                                                              \
    impl::NAMESPACE::OPERATOR_NAME##_kernel<float>(                             \
      float_output.data_ptr<float>(),                                           \
      numel                                                                     \
      TORCHSCIENCE_COMMA_IF(ARGS)                                               \
      TORCHSCIENCE_UNPACK(ARGS)                                                 \
    );                                                                          \
  }                                                                             \
                                                                                \
  at::Tensor output = at::quantize_per_tensor(                                  \
    float_output, scale, zero_point, qtype                                      \
  );                                                                            \
                                                                                \
  if (requires_grad) {                                                          \
    TORCH_WARN(#OPERATOR_NAME ": requires_grad ignored for quantized tensor");  \
  }                                                                             \
                                                                                \
  return output;                                                                \
}                                                                               \
                                                                                \
}  /* namespace torchscience::quantized::cpu::NAMESPACE */                      \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module) {                        \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::quantized::cpu::NAMESPACE::OPERATOR_NAME                     \
  );                                                                            \
}
