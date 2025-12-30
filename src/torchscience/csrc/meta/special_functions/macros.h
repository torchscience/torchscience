#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

#define TORCHSCIENCE_META_UNARY_OPERATOR(name, arg)                            \
namespace torchscience::meta::special_functions {                              \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg##_input                                                \
) {                                                                            \
  at::Tensor output;                                                           \
                                                                               \
  return at::TensorIteratorConfig()                                            \
    .add_output(output)                                                        \
    .add_const_input(arg##_input)                                              \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build()                                                                   \
    .output();                                                                 \
}                                                                              \
                                                                               \
inline at::Tensor name##_backward(                                             \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg##_input                                                \
) {                                                                            \
  at::Tensor gradient_output;                                                  \
                                                                               \
  return at::TensorIteratorConfig()                                            \
    .add_output(gradient_output)                                               \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg##_input)                                              \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build()                                                                   \
    .output();                                                                 \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(            \
  const at::Tensor &gradient_gradient_input,                                   \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg##_input                                                \
) {                                                                            \
  if (!gradient_gradient_input.defined()) {                                    \
    return {};                                                                 \
  }                                                                            \
                                                                               \
  at::Tensor gradient_gradient_output;                                         \
  at::Tensor gradient_output;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_gradient_output)                                      \
    .add_output(gradient_output)                                               \
    .add_const_input(gradient_gradient_input)                                  \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg##_input)                                              \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::meta::special_functions */                        \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                               \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::meta::special_functions::name                                \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::meta::special_functions::name##_backward                     \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::meta::special_functions::name##_backward_backward            \
  );                                                                           \
}
