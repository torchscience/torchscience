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

#define TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL(SCHEMA_NAME, ARG0, ARG1, ARG2) \
  at::Tensor SCHEMA_NAME##_forward(                                             \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2                                                      \
  ) {                                                                           \
    auto dequantized_##ARG0 = ARG0.dequantize();                                \
    auto dequantized_##ARG1 = ARG1.dequantize();                                \
    auto dequantized_##ARG2 = ARG2.dequantize();                                \
                                                                                \
    at::Tensor output = at::empty_like(dequantized_##ARG0);                     \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
        .add_output(output)                                                     \
        .add_input(dequantized_##ARG0)                                          \
        .add_input(dequantized_##ARG1)                                          \
        .add_input(dequantized_##ARG2)                                          \
        .build();                                                               \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES(                                                 \
        iterator.common_dtype(),                                                \
        #SCHEMA_NAME,                                                           \
        [&]() {                                                                 \
          at::native::cpu_kernel(                                               \
              iterator,                                                         \
              [](scalar_t ARG0, scalar_t ARG1, scalar_t ARG2) -> scalar_t {     \
                return SCHEMA_NAME(ARG0, ARG1, ARG2);                           \
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

#define TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL_IMPL(SCHEMA_NAME)             \
  TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module) {                      \
    module.impl(                                                                \
        "_" #SCHEMA_NAME,                                                       \
        &SCHEMA_NAME##_forward                                                  \
    );                                                                          \
  }

#define TORCHSCIENCE_QUATERNARY_QUANTIZED_CPU_KERNEL(SCHEMA_NAME, ARG0, ARG1, ARG2, ARG3) \
  at::Tensor SCHEMA_NAME##_forward(                                             \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3                                                      \
  ) {                                                                           \
    auto dequantized_##ARG0 = ARG0.dequantize();                                \
    auto dequantized_##ARG1 = ARG1.dequantize();                                \
    auto dequantized_##ARG2 = ARG2.dequantize();                                \
    auto dequantized_##ARG3 = ARG3.dequantize();                                \
                                                                                \
    at::Tensor output = at::empty_like(dequantized_##ARG0);                     \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
        .add_output(output)                                                     \
        .add_input(dequantized_##ARG0)                                          \
        .add_input(dequantized_##ARG1)                                          \
        .add_input(dequantized_##ARG2)                                          \
        .add_input(dequantized_##ARG3)                                          \
        .build();                                                               \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES(                                                 \
        iterator.common_dtype(),                                                \
        #SCHEMA_NAME,                                                           \
        [&]() {                                                                 \
          at::native::cpu_kernel(                                               \
              iterator,                                                         \
              [](scalar_t ARG0, scalar_t ARG1, scalar_t ARG2,                   \
                 scalar_t ARG3) -> scalar_t {                                   \
                return SCHEMA_NAME(ARG0, ARG1, ARG2, ARG3);                     \
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

#define TORCHSCIENCE_QUATERNARY_QUANTIZED_CPU_KERNEL_IMPL(SCHEMA_NAME)          \
  TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module) {                      \
    module.impl(                                                                \
        "_" #SCHEMA_NAME,                                                       \
        &SCHEMA_NAME##_forward                                                  \
    );                                                                          \
  }

#define TORCHSCIENCE_QUINARY_QUANTIZED_CPU_KERNEL(SCHEMA_NAME, ARG0, ARG1, ARG2, ARG3, ARG4) \
  at::Tensor SCHEMA_NAME##_forward(                                             \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3,                                                     \
    const at::Tensor& ARG4                                                      \
  ) {                                                                           \
    auto dequantized_##ARG0 = ARG0.dequantize();                                \
    auto dequantized_##ARG1 = ARG1.dequantize();                                \
    auto dequantized_##ARG2 = ARG2.dequantize();                                \
    auto dequantized_##ARG3 = ARG3.dequantize();                                \
    auto dequantized_##ARG4 = ARG4.dequantize();                                \
                                                                                \
    at::Tensor output = at::empty_like(dequantized_##ARG0);                     \
                                                                                \
    auto iterator = at::TensorIteratorConfig()                                  \
        .add_output(output)                                                     \
        .add_input(dequantized_##ARG0)                                          \
        .add_input(dequantized_##ARG1)                                          \
        .add_input(dequantized_##ARG2)                                          \
        .add_input(dequantized_##ARG3)                                          \
        .add_input(dequantized_##ARG4)                                          \
        .build();                                                               \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES(                                                 \
        iterator.common_dtype(),                                                \
        #SCHEMA_NAME,                                                           \
        [&]() {                                                                 \
          at::native::cpu_kernel(                                               \
              iterator,                                                         \
              [](scalar_t ARG0, scalar_t ARG1, scalar_t ARG2,                   \
                 scalar_t ARG3, scalar_t ARG4) -> scalar_t {                    \
                return SCHEMA_NAME(ARG0, ARG1, ARG2, ARG3, ARG4);               \
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

#define TORCHSCIENCE_QUINARY_QUANTIZED_CPU_KERNEL_IMPL(SCHEMA_NAME)             \
  TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module) {                      \
    module.impl(                                                                \
        "_" #SCHEMA_NAME,                                                       \
        &SCHEMA_NAME##_forward                                                  \
    );                                                                          \
  }
