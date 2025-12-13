#pragma once

#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <torch/torch.h>
#include <thrust/tuple.h>

#define TORCHSCIENCE_UNARY_CUDA_KERNEL(SCHEMA_NAME)                             \
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
    AT_DISPATCH_FLOATING_TYPES(                                     \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME,                                                             \
      [&]() {                                                                   \
        at::native::gpu_kernel(                                                 \
          iterator,                                                             \
          []GPU_LAMBDA(                                                         \
            scalar_t x                                                          \
          ) -> scalar_t {                                                       \
            return SCHEMA_NAME(                                                  \
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
    AT_DISPATCH_FLOATING_TYPES(                                     \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME "_backward",                                                 \
      [&]() {                                                                   \
        at::native::gpu_kernel(                                                 \
          iterator,                                                             \
          []GPU_LAMBDA(                                                         \
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

#define TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(SCHEMA_NAME)                        \
  TORCH_LIBRARY_IMPL(                                                           \
    torchscience,                                                               \
    CUDA,                                                                       \
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

#define TORCHSCIENCE_BINARY_CUDA_KERNEL(SCHEMA_NAME, ARG0, ARG1)                \
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
    AT_DISPATCH_FLOATING_TYPES(                                                 \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME,                                                             \
      [&]() {                                                                   \
        at::native::gpu_kernel(                                                 \
          iterator,                                                             \
          []GPU_LAMBDA(                                                         \
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
    AT_DISPATCH_FLOATING_TYPES(                                                 \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME "_backward",                                                 \
      [&]() {                                                                   \
        at::native::gpu_kernel_multiple_outputs(                                \
          iterator,                                                             \
          []GPU_LAMBDA(                                                         \
            scalar_t gradient,                                                  \
            scalar_t ARG0,                                                      \
            scalar_t ARG1                                                       \
          ) -> thrust::tuple<scalar_t, scalar_t> {                              \
            auto [grad_##ARG0, grad_##ARG1] = SCHEMA_NAME##_backward(           \
              ARG0,                                                             \
              ARG1                                                              \
            );                                                                  \
            return thrust::make_tuple(                                          \
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

#define TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(SCHEMA_NAME)                       \
  TORCH_LIBRARY_IMPL(                                                           \
    torchscience,                                                               \
    CUDA,                                                                       \
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

#define TORCHSCIENCE_TERNARY_CUDA_KERNEL(SCHEMA_NAME, ARG0, ARG1, ARG2)         \
  at::Tensor SCHEMA_NAME##_forward_kernel(                                      \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2                                                      \
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
      .add_input(                                                               \
        ARG2                                                                    \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES(                                                 \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME,                                                             \
      [&]() {                                                                   \
        at::native::gpu_kernel(                                                 \
          iterator,                                                             \
          []GPU_LAMBDA(                                                         \
            scalar_t ARG0,                                                      \
            scalar_t ARG1,                                                      \
            scalar_t ARG2                                                       \
          ) -> scalar_t {                                                       \
            return SCHEMA_NAME(                                                 \
              ARG0,                                                             \
              ARG1,                                                             \
              ARG2                                                              \
            );                                                                  \
          }                                                                     \
        );                                                                      \
      }                                                                         \
    );                                                                          \
                                                                                \
    return output;                                                              \
  }                                                                             \
                                                                                \
  std::tuple<at::Tensor, at::Tensor, at::Tensor>                                \
  SCHEMA_NAME##_backward_kernel(                                                \
    const at::Tensor& gradient_output,                                          \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2                                                      \
  ) {                                                                           \
    at::Tensor grad_##ARG0 = at::empty_like(ARG0);                              \
    at::Tensor grad_##ARG1 = at::empty_like(ARG1);                              \
    at::Tensor grad_##ARG2 = at::empty_like(ARG2);                              \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
      .add_output(                                                              \
        grad_##ARG0                                                             \
      )                                                                         \
      .add_output(                                                              \
        grad_##ARG1                                                             \
      )                                                                         \
      .add_output(                                                              \
        grad_##ARG2                                                             \
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
      .add_input(                                                               \
        ARG2                                                                    \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES(                                                 \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME "_backward",                                                 \
      [&]() {                                                                   \
        at::native::gpu_kernel_multiple_outputs(                                \
          iterator,                                                             \
          []GPU_LAMBDA(                                                         \
            scalar_t gradient,                                                  \
            scalar_t ARG0,                                                      \
            scalar_t ARG1,                                                      \
            scalar_t ARG2                                                       \
          ) -> thrust::tuple<scalar_t, scalar_t, scalar_t> {                    \
            auto [grad_##ARG0, grad_##ARG1, grad_##ARG2] =                       \
              SCHEMA_NAME##_backward(                                           \
                ARG0,                                                           \
                ARG1,                                                           \
                ARG2                                                            \
              );                                                                \
            return thrust::make_tuple(                                          \
              gradient * grad_##ARG0,                                           \
              gradient * grad_##ARG1,                                           \
              gradient * grad_##ARG2                                            \
            );                                                                  \
          }                                                                     \
        );                                                                      \
      }                                                                         \
    );                                                                          \
                                                                                \
    return std::make_tuple(grad_##ARG0, grad_##ARG1, grad_##ARG2);              \
  }

#define TORCHSCIENCE_TERNARY_CUDA_KERNEL_IMPL(SCHEMA_NAME)                      \
  TORCH_LIBRARY_IMPL(                                                           \
    torchscience,                                                               \
    CUDA,                                                                       \
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

