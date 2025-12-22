#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../core/creation_common.h"

#ifndef TORCHSCIENCE_UNPACK_IMPL
#define TORCHSCIENCE_UNPACK_IMPL(...) __VA_ARGS__
#define TORCHSCIENCE_UNPACK(X) TORCHSCIENCE_UNPACK_IMPL X
#define TORCHSCIENCE_COMMA_IF_IMPL(...) __VA_OPT__(,)
#define TORCHSCIENCE_COMMA_IF(X) TORCHSCIENCE_COMMA_IF_IMPL(TORCHSCIENCE_UNPACK(X))
#endif

// Meta tensors: shape inference only, no actual memory allocation
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
  std::vector<int64_t> shape_vec = OUTPUT_SHAPE;                                \
  ::torchscience::core::check_size_nonnegative(shape_vec, #OPERATOR_NAME);      \
                                                                                \
  /* Meta tensors always use kMeta device, ignore device parameter */           \
  auto options = at::TensorOptions()                                            \
    .dtype(dtype.value_or(                                                      \
      c10::typeMetaToScalarType(at::get_default_dtype())                        \
    ))                                                                          \
    .layout(layout.value_or(at::kStrided))                                      \
    .device(at::kMeta)                                                          \
    .requires_grad(requires_grad);                                              \
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
