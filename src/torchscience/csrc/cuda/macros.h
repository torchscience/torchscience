#pragma once

#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <torch/torch.h>

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
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
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
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
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

#define TORCHSCIENCE_BINARY_CUDA_KERNEL(SCHEMA_NAME, ARG1, ARG2)                \
  at::Tensor SCHEMA_NAME##_forward_kernel(                                      \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2                                                      \
  ) {                                                                           \
    at::Tensor output = at::empty_like(                                         \
      ARG1                                                                      \
    );                                                                          \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
      .add_output(                                                              \
        output                                                                  \
      )                                                                         \
      .add_input(                                                               \
        ARG1                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG2                                                                    \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME,                                                             \
      [&]() {                                                                   \
        at::native::gpu_kernel(                                                 \
          iterator,                                                             \
          []GPU_LAMBDA(                                                         \
            scalar_t ARG1,                                                      \
            scalar_t ARG2                                                       \
          ) -> scalar_t {                                                       \
            return SCHEMA_NAME(                                                  \
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
  at::Tensor SCHEMA_NAME##_backward_kernel(                                     \
    const at::Tensor& gradient_output,                                          \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2                                                      \
  ) {                                                                           \
    at::Tensor gradient_##ARG1 = at::empty_like(                                \
      ARG1                                                                      \
    );                                                                          \
                                                                                \
    at::Tensor gradient_##ARG2 = at::empty_like(                                \
      ARG2                                                                      \
    );                                                                          \
                                                                                \
    auto iterator_##ARG1 = at::TensorIteratorConfig()                           \
      .add_output(                                                              \
        gradient_##ARG1                                                         \
      )                                                                         \
      .add_input(                                                               \
        gradient_output                                                         \
      )                                                                         \
      .add_input(                                                               \
        ARG1                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG2                                                                    \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
      iterator_##ARG1.common_dtype(),                                           \
      #SCHEMA_NAME "_backward_" #ARG1,                                          \
      [&]() {                                                                   \
        at::native::gpu_kernel(                                                 \
          iterator_##ARG1,                                                      \
          []GPU_LAMBDA(                                                         \
            scalar_t gradient,                                                  \
            scalar_t ARG1,                                                      \
            scalar_t ARG2                                                       \
          ) -> scalar_t {                                                       \
            return gradient * SCHEMA_NAME##_backward_##ARG1(                    \
              ARG1,                                                             \
              ARG2                                                              \
            );                                                                  \
          }                                                                     \
        );                                                                      \
      }                                                                         \
    );                                                                          \
                                                                                \
    auto iterator_##ARG2 = at::TensorIteratorConfig()                           \
      .add_output(                                                              \
        gradient_##ARG2                                                         \
      )                                                                         \
      .add_input(                                                               \
        gradient_output                                                         \
      )                                                                         \
      .add_input(                                                               \
        ARG1                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG2                                                                    \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
      iterator_##ARG2.common_dtype(),                                           \
      #SCHEMA_NAME "_backward_" #ARG2,                                          \
      [&]() {                                                                   \
        at::native::gpu_kernel(                                                 \
          iterator_##ARG2,                                                      \
          []GPU_LAMBDA(                                                         \
            scalar_t gradient,                                                  \
            scalar_t ARG1,                                                      \
            scalar_t ARG2                                                       \
          ) -> scalar_t {                                                       \
            return gradient * SCHEMA_NAME##_backward_##ARG2(                    \
              ARG1,                                                             \
              ARG2                                                              \
            );                                                                  \
          }                                                                     \
        );                                                                      \
      }                                                                         \
    );                                                                          \
                                                                                \
    return at::stack(                                                           \
      {                                                                         \
        gradient_##ARG1,                                                        \
        gradient_##ARG2                                                         \
      },                                                                        \
      0                                                                         \
    );                                                                          \
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

#define TORCHSCIENCE_TERNARY_CUDA_KERNEL(SCHEMA_NAME, ARG1, ARG2, ARG3)         \
  at::Tensor SCHEMA_NAME##_forward_kernel(                                      \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3                                                      \
  ) {                                                                           \
    at::Tensor output = at::empty_like(                                         \
      ARG1                                                                      \
    );                                                                          \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
      .add_output(                                                              \
        output                                                                  \
      )                                                                         \
      .add_input(                                                               \
        ARG1                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG2                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG3                                                                    \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
      iterator.common_dtype(),                                                  \
      #SCHEMA_NAME,                                                             \
      [&]() {                                                                   \
        at::native::gpu_kernel(                                                 \
          iterator,                                                             \
          []GPU_LAMBDA(                                                         \
            scalar_t ARG1,                                                      \
            scalar_t ARG2,                                                      \
            scalar_t ARG3                                                       \
          ) -> scalar_t {                                                       \
            return SCHEMA_NAME(                                                  \
              ARG1,                                                             \
              ARG2,                                                             \
              ARG3                                                              \
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
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3                                                      \
  ) {                                                                           \
    at::Tensor gradient_##ARG1 = at::empty_like(                                \
      ARG1                                                                      \
    );                                                                          \
                                                                                \
    at::Tensor gradient_##ARG2 = at::empty_like(                                \
      ARG2                                                                      \
    );                                                                          \
                                                                                \
    at::Tensor gradient_##ARG3 = at::empty_like(                                \
      ARG3                                                                      \
    );                                                                          \
                                                                                \
    auto iterator_##ARG1 = at::TensorIteratorConfig()                           \
      .add_output(                                                              \
        gradient_##ARG1                                                         \
      )                                                                         \
      .add_input(                                                               \
        gradient_output                                                         \
      )                                                                         \
      .add_input(                                                               \
        ARG1                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG2                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG3                                                                    \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
      iterator_##ARG1.common_dtype(),                                           \
      #SCHEMA_NAME "_backward_" #ARG1,                                          \
      [&]() {                                                                   \
        at::native::gpu_kernel(                                                 \
          iterator_##ARG1,                                                      \
          []GPU_LAMBDA(                                                         \
            scalar_t gradient,                                                  \
            scalar_t ARG1,                                                      \
            scalar_t ARG2,                                                      \
            scalar_t ARG3                                                       \
          ) -> scalar_t {                                                       \
            return gradient * SCHEMA_NAME##_backward_##ARG1(                    \
              ARG1,                                                             \
              ARG2,                                                             \
              ARG3                                                              \
            );                                                                  \
          }                                                                     \
        );                                                                      \
      }                                                                         \
    );                                                                          \
                                                                                \
    auto iterator_##ARG2 = at::TensorIteratorConfig()                           \
      .add_output(                                                              \
        gradient_##ARG2                                                         \
      )                                                                         \
      .add_input(                                                               \
        gradient_output                                                         \
      )                                                                         \
      .add_input(                                                               \
        ARG1                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG2                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG3                                                                    \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
      iterator_##ARG2.common_dtype(),                                           \
      #SCHEMA_NAME "_backward_" #ARG2,                                          \
      [&]() {                                                                   \
        at::native::gpu_kernel(                                                 \
          iterator_##ARG2,                                                      \
          []GPU_LAMBDA(                                                         \
            scalar_t gradient,                                                  \
            scalar_t ARG1,                                                      \
            scalar_t ARG2,                                                      \
            scalar_t ARG3                                                       \
          ) -> scalar_t {                                                       \
            return gradient * SCHEMA_NAME##_backward_##ARG2(                    \
              ARG1,                                                             \
              ARG2,                                                             \
              ARG3                                                              \
            );                                                                  \
          }                                                                     \
        );                                                                      \
      }                                                                         \
    );                                                                          \
                                                                                \
    auto iterator_##ARG3 = at::TensorIteratorConfig()                           \
      .add_output(                                                              \
        gradient_##ARG3                                                         \
      )                                                                         \
      .add_input(                                                               \
        gradient_output                                                         \
      )                                                                         \
      .add_input(                                                               \
        ARG1                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG2                                                                    \
      )                                                                         \
      .add_input(                                                               \
        ARG3                                                                    \
      )                                                                         \
      .build();                                                                 \
                                                                                \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(                                     \
      iterator_##ARG3.common_dtype(),                                           \
      #SCHEMA_NAME "_backward_" #ARG3,                                          \
      [&]() {                                                                   \
        at::native::gpu_kernel(                                                 \
          iterator_##ARG3,                                                      \
          []GPU_LAMBDA(                                                         \
            scalar_t gradient,                                                  \
            scalar_t ARG1,                                                      \
            scalar_t ARG2,                                                      \
            scalar_t ARG3                                                       \
          ) -> scalar_t {                                                       \
            return gradient * SCHEMA_NAME##_backward_##ARG3(                    \
              ARG1,                                                             \
              ARG2,                                                             \
              ARG3                                                              \
            );                                                                  \
          }                                                                     \
        );                                                                      \
      }                                                                         \
    );                                                                          \
                                                                                \
    return at::stack(                                                           \
      {                                                                         \
        gradient_##ARG1,                                                        \
        gradient_##ARG2,                                                        \
        gradient_##ARG3                                                         \
      },                                                                        \
      0                                                                         \
    );                                                                          \
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
