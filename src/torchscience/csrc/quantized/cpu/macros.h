#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#define TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(SCHEMA_NAME)                    \
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
          at::native::cpu_kernel(                                               \
              iterator,                                                         \
              [](scalar_t x) -> scalar_t {                                      \
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

#define TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(SCHEMA_NAME)               \
  TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module) {                      \
    module.impl(                                                                \
        "_" #SCHEMA_NAME,                                                       \
        &SCHEMA_NAME##_forward                                                  \
    );                                                                          \
  }

#define TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(SCHEMA_NAME, ARG0, ARG1)       \
  at::Tensor SCHEMA_NAME##_forward(                                             \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1                                                      \
  ) {                                                                           \
    auto dequantized_##ARG0 = ARG0.dequantize();                                \
    auto dequantized_##ARG1 = ARG1.dequantize();                                \
                                                                                \
    at::Tensor output = at::empty_like(dequantized_##ARG0);                     \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
        .add_output(output)                                                     \
        .add_input(dequantized_##ARG0)                                          \
        .add_input(dequantized_##ARG1)                                          \
        .build();                                                               \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES(                                                 \
        iterator.common_dtype(),                                                \
        #SCHEMA_NAME,                                                           \
        [&]() {                                                                 \
          at::native::cpu_kernel(                                               \
              iterator,                                                         \
              [](scalar_t ARG0, scalar_t ARG1) -> scalar_t {                    \
                return SCHEMA_NAME(ARG0, ARG1);                                 \
              }                                                                 \
          );                                                                    \
        }                                                                       \
    );                                                                          \
                                                                                \
    return at::quantize_per_tensor(                                             \
        output,                                                                 \
        ARG0.q_scale(),                                                         \
        ARG0.q_zero_point(),                                                    \
        ARG0.scalar_type()                                                      \
    );                                                                          \
  }

#define TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(SCHEMA_NAME)              \
  TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module) {                      \
    module.impl(                                                                \
        "_" #SCHEMA_NAME,                                                       \
        &SCHEMA_NAME##_forward                                                  \
    );                                                                          \
  }
