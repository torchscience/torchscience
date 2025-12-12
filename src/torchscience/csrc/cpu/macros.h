#pragma once

#include <ATen/Dispatch.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/TensorIterator.h>
#include <torch/torch.h>

#define TORCHSCIENCE_UNARY_CPU_KERNEL(SCHEMA_NAME)                              \
  at::Tensor SCHEMA_NAME##_forward_kernel(                                      \
    const at::Tensor& input                                                     \
  ) {                                                                           \
    at::Tensor output = at::empty_like(input);                                  \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
      .add_output(                                                              \
        output                                                                  \
      )                                                                         \
      .add_input(                                                               \
        input                                                                   \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME,                                                             \
      [&]() {                                                                   \
        at::native::cpu_kernel(                                                 \
          iterator,                                                             \
          [](                                                                   \
            scalar_t x                                                          \
          ) -> scalar_t {                                                       \
            return SCHEMA_NAME(                                                 \
              x                                                                 \
            );                                                                  \
          }                                                                     \
        );                                                                      \
      }                                                                         \
    );                                                                          \
                                                                                \
    return output;                                                              \
  }                                                                             \
                                                                                \
  at::Tensor SCHEMA_NAME##_backward_kernel(                                     \
    const at::Tensor& gradient_output,                                          \
    const at::Tensor& input                                                     \
  ) {                                                                           \
    at::Tensor output = at::empty_like(input);                                  \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
      .add_output(                                                              \
        output                                                                  \
      )                                                                         \
      .add_input(                                                               \
        gradient_output                                                         \
      )                                                                         \
      .add_input(                                                               \
        input                                                                   \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME "_backward",                                                 \
      [&]() {                                                                   \
        at::native::cpu_kernel(                                                 \
          iterator,                                                             \
          [](                                                                   \
            scalar_t gradient,                                                  \
            scalar_t x                                                          \
          ) -> scalar_t {                                                       \
            return gradient * SCHEMA_NAME##_backward(                           \
              x                                                                 \
            );                                                                  \
          }                                                                     \
        );                                                                      \
      }                                                                         \
    );                                                                          \
                                                                                \
    return output;                                                              \
  }

#define TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(SCHEMA_NAME)                         \
  TORCH_LIBRARY_IMPL(                                                           \
    torchscience,                                                               \
    CPU,                                                                        \
    module                                                                      \
  ) {                                                                           \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME##_forward_kernel                                             \
    );                                                                          \
                                                                                \
    module.impl(                                                                \
      "_" #SCHEMA_NAME "_backward",                                             \
      &SCHEMA_NAME##_backward_kernel                                            \
    );                                                                          \
  }

#define TORCHSCIENCE_BINARY_CPU_KERNEL(SCHEMA_NAME, ARG0, ARG1)                 \
  at::Tensor SCHEMA_NAME##_forward_kernel(                                      \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1                                                      \
  ) {                                                                           \
    at::Tensor output = at::empty_like(ARG0);                                   \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
      .add_output(                                                              \
        output                                                                  \
      )                                                                         \
      .add_input(                                                               \
        ARG0                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG1                                                                    \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME,                                                             \
      [&]() {                                                                   \
        at::native::cpu_kernel(                                                 \
          iterator,                                                             \
          [](                                                                   \
            scalar_t ARG0,                                                      \
            scalar_t ARG1                                                       \
          ) -> scalar_t {                                                       \
            return SCHEMA_NAME(                                                 \
              ARG0,                                                             \
              ARG1                                                              \
            );                                                                  \
          }                                                                     \
        );                                                                      \
      }                                                                         \
    );                                                                          \
                                                                                \
    return output;                                                              \
  }                                                                             \
                                                                                \
  std::tuple<at::Tensor, at::Tensor> SCHEMA_NAME##_backward_kernel(             \
    const at::Tensor& gradient_output,                                          \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1                                                      \
  ) {                                                                           \
    at::Tensor grad_##ARG0 = at::empty_like(ARG0);                              \
    at::Tensor grad_##ARG1 = at::empty_like(ARG1);                              \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
      .add_output(                                                              \
        grad_##ARG0                                                             \
      )                                                                         \
      .add_output(                                                              \
        grad_##ARG1                                                             \
      )                                                                         \
      .add_input(                                                               \
        gradient_output                                                         \
      )                                                                         \
      .add_input(                                                               \
        ARG0                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG1                                                                    \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME "_backward",                                                 \
      [&]() {                                                                   \
        at::native::cpu_kernel_multiple_outputs(                                \
          iterator,                                                             \
          [](                                                                   \
            scalar_t gradient,                                                  \
            scalar_t ARG0,                                                      \
            scalar_t ARG1                                                       \
          ) -> std::tuple<scalar_t, scalar_t> {                                 \
            auto [grad_##ARG0, grad_##ARG1] = SCHEMA_NAME##_backward(           \
              ARG0,                                                             \
              ARG1                                                              \
            );                                                                  \
            return std::make_tuple(                                             \
              gradient * grad_##ARG0,                                           \
              gradient * grad_##ARG1                                            \
            );                                                                  \
          }                                                                     \
        );                                                                      \
      }                                                                         \
    );                                                                          \
                                                                                \
    return std::make_tuple(grad_##ARG0, grad_##ARG1);                           \
  }

#define TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(SCHEMA_NAME)                        \
  TORCH_LIBRARY_IMPL(                                                           \
    torchscience,                                                               \
    CPU,                                                                        \
    module                                                                      \
  ) {                                                                           \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME##_forward_kernel                                             \
    );                                                                          \
                                                                                \
    module.impl(                                                                \
      "_" #SCHEMA_NAME "_backward",                                             \
      &SCHEMA_NAME##_backward_kernel                                            \
    );                                                                          \
  }
