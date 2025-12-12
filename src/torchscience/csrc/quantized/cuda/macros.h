#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#define TORCHSCIENCE_UNARY_QUANTIZED_CUDA_KERNEL(SCHEMA_NAME)                   \
  at::Tensor SCHEMA_NAME##_forward(const at::Tensor& input) {                   \
    auto dequantized = input.dequantize();                                      \
                                                                                \
    at::Tensor output = at::empty_like(dequantized);                            \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
        .add_output(output)                                                     \
        .add_input(dequantized)                                                 \
        .build();                                                               \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES(                                                 \
        iterator.common_dtype(),                                                \
        #SCHEMA_NAME,                                                           \
        [&]() {                                                                 \
          at::native::gpu_kernel(                                               \
              iterator,                                                         \
              []GPU_LAMBDA(scalar_t x) -> scalar_t {                            \
                return SCHEMA_NAME(x);                                          \
              }                                                                 \
          );                                                                    \
        }                                                                       \
    );                                                                          \
                                                                                \
    return at::quantize_per_tensor(                                             \
        output,                                                                 \
        input.q_scale(),                                                        \
        input.q_zero_point(),                                                   \
        input.scalar_type()                                                     \
    );                                                                          \
  }

#define TORCHSCIENCE_UNARY_QUANTIZED_CUDA_KERNEL_IMPL(SCHEMA_NAME)              \
  TORCH_LIBRARY_IMPL(torchscience, QuantizedCUDA, module) {                     \
    module.impl(                                                                \
        "_" #SCHEMA_NAME,                                                       \
        &SCHEMA_NAME##_forward                                                  \
    );                                                                          \
  }
