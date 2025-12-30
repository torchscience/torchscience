#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

#define TORCHSCIENCE_CPU_UNARY_OPERATOR(name, arg)                             \
namespace torchscience::cpu::special_functions {                               \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg##_input                                                \
) {                                                                            \
  at::Tensor output;                                                           \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(output)                                                        \
    .add_const_input(arg##_input)                                              \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                             \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name,                                                                     \
    [&] {                                                                      \
      at::native::cpu_kernel(                                                  \
        iterator,                                                              \
        [] (                                                                   \
          scalar_t arg                                                         \
        ) -> scalar_t {                                                        \
          return kernel::special_functions::name(                              \
            arg                                                                \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return iterator.output();                                                    \
}                                                                              \
                                                                               \
inline at::Tensor name##_backward(                                             \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg##_input                                                \
) {                                                                            \
  at::Tensor gradient_output;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_output)                                               \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg##_input)                                              \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                             \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward",                                                         \
    [&] {                                                                      \
      at::native::cpu_kernel(                                                  \
        iterator,                                                              \
        [] (                                                                   \
          scalar_t gradient,                                                   \
          scalar_t arg                                                         \
        ) -> scalar_t {                                                        \
          return kernel::special_functions::name##_backward(                   \
            gradient,                                                          \
            arg                                                                \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return iterator.output();                                                    \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(            \
  const at::Tensor &gradient_gradient_input,                                   \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg##_input                                                \
) {                                                                            \
  if (!gradient_gradient_input.defined()) {                                    \
    return {                                                                   \
      at::Tensor(),                                                            \
      at::Tensor()                                                             \
    };                                                                         \
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
  AT_DISPATCH_FLOATING_TYPES_AND2(                                             \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward_backward",                                                \
    [&] {                                                                      \
      at::native::cpu_kernel_multiple_outputs(                                 \
        iterator,                                                              \
        [](                                                                    \
          scalar_t gradient_gradient,                                          \
          scalar_t gradient,                                                   \
          scalar_t arg                                                         \
        ) -> std::tuple<                                                       \
          scalar_t,                                                            \
          scalar_t                                                             \
        > {                                                                    \
          return kernel::special_functions::name##_backward_backward(          \
            gradient_gradient,                                                 \
            gradient,                                                          \
            arg                                                                \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::cpu::special_functions */                         \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                                \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::cpu::special_functions::name                                 \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::cpu::special_functions::name##_backward                      \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::cpu::special_functions::name##_backward_backward             \
  );                                                                           \
}
