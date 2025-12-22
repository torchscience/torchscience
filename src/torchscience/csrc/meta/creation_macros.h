#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

// =============================================================================
// Helper macros for handling parenthesized parameter lists
// =============================================================================

#ifndef TORCHSCIENCE_UNPACK_IMPL
#define TORCHSCIENCE_UNPACK_IMPL(...) __VA_ARGS__
#define TORCHSCIENCE_UNPACK(X) TORCHSCIENCE_UNPACK_IMPL X
#define TORCHSCIENCE_COMMA_IF_IMPL(...) __VA_OPT__(,)
#define TORCHSCIENCE_COMMA_IF(X) TORCHSCIENCE_COMMA_IF_IMPL(TORCHSCIENCE_UNPACK(X))
#endif

// =============================================================================
// META_CREATION_OPERATOR
// =============================================================================
// Flexible macro for shape inference of creation operators on Meta device.
// No computation performed - just returns a meta tensor with correct shape.
//
// Parameters:
//   NAMESPACE     - Namespace (e.g., window_function)
//   OPERATOR_NAME - Operator name (e.g., rectangular_window)
//   OUTPUT_SHAPE  - Shape expression (e.g., {n})
//   PARAMS        - Parenthesized typed params: (int64_t n) or ()
//
// Usage:
//   META_CREATION_OPERATOR(window_function, rectangular_window, {n}, (int64_t n))
// =============================================================================

#define META_CREATION_OPERATOR(                                                 \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  OUTPUT_SHAPE,                                                                 \
  PARAMS                                                                        \
)                                                                               \
                                                                                \
namespace torchscience::meta::NAMESPACE {                                       \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  TORCHSCIENCE_UNPACK(PARAMS)                                                   \
  TORCHSCIENCE_COMMA_IF(PARAMS)                                                 \
  const c10::optional<at::ScalarType>& dtype,                                   \
  const c10::optional<at::Layout>& layout,                                      \
  const c10::optional<at::Device>& device,                                      \
  bool requires_grad                                                            \
) {                                                                             \
  auto options = at::TensorOptions()                                            \
    .dtype(dtype.value_or(                                                      \
      c10::typeMetaToScalarType(at::get_default_dtype())                        \
    ))                                                                          \
    .layout(layout.value_or(at::kStrided))                                      \
    .device(at::kMeta)                                                          \
    .requires_grad(requires_grad);                                              \
                                                                                \
  std::vector<int64_t> shape_vec = OUTPUT_SHAPE;                                \
  for (auto s : shape_vec) {                                                    \
    TORCH_CHECK(s >= 0,                                                         \
      #OPERATOR_NAME ": size must be non-negative, got ", s);                   \
  }                                                                             \
                                                                                \
  return at::empty(shape_vec, options);                                         \
}                                                                               \
                                                                                \
}  /* namespace torchscience::meta::NAMESPACE */                                \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                                \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME                               \
  );                                                                            \
}
