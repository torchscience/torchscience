#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#define TORCHSCIENCE_UNARY_SPARSE_CSR_CUDA_KERNEL(SCHEMA_NAME)                  \
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

#define TORCHSCIENCE_UNARY_SPARSE_CSR_CUDA_KERNEL_IMPL(SCHEMA_NAME)             \
  TORCH_LIBRARY_IMPL(torchscience, SparseCsrCUDA, module) {                     \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME##_forward_kernel                                             \
    );                                                                          \
  }
