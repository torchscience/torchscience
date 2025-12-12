#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#define TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(SCHEMA_NAME)                   \
  at::Tensor SCHEMA_NAME##_forward_kernel(const at::Tensor& input) {            \
    auto values = input.values();                                               \
                                                                                \
    auto output = c10::Dispatcher::singleton()                                  \
      .findSchemaOrThrow(                                                       \
        "torchscience::_" #SCHEMA_NAME,                                         \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&                                                       \
      )>()                                                                      \
      .call(                                                                    \
        values                                                                  \
      );                                                                        \
                                                                                \
    return at::sparse_csr_tensor(                                               \
      input.crow_indices(),                                                     \
      input.col_indices(),                                                      \
      output,                                                                   \
      input.sizes(),                                                            \
      input.options()                                                           \
    );                                                                          \
  }

#define TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(SCHEMA_NAME)              \
  TORCH_LIBRARY_IMPL(torchscience, SparseCsrCPU, module) {                      \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME##_forward_kernel                                             \
    );                                                                          \
  }

#define TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(SCHEMA_NAME, ARG0, ARG1)      \
  at::Tensor SCHEMA_NAME##_forward_kernel(                                      \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1                                                      \
  ) {                                                                           \
    auto values_##ARG0 = ARG0.values();                                         \
    auto values_##ARG1 = ARG1.values();                                         \
                                                                                \
    auto output = c10::Dispatcher::singleton()                                  \
      .findSchemaOrThrow(                                                       \
        "torchscience::_" #SCHEMA_NAME,                                         \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>()                                                                      \
      .call(                                                                    \
        values_##ARG0,                                                          \
        values_##ARG1                                                           \
      );                                                                        \
                                                                                \
    return at::sparse_csr_tensor(                                               \
      ARG0.crow_indices(),                                                      \
      ARG0.col_indices(),                                                       \
      output,                                                                   \
      ARG0.sizes(),                                                             \
      ARG0.options()                                                            \
    );                                                                          \
  }

#define TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(SCHEMA_NAME)             \
  TORCH_LIBRARY_IMPL(torchscience, SparseCsrCPU, module) {                      \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME##_forward_kernel                                             \
    );                                                                          \
  }
