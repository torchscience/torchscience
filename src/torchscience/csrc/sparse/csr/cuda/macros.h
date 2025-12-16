#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

#define SPARSE_CSR_CUDA_UNARY_OPERATOR(                                         \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::sparse::csr::cuda::NAMESPACE {                          \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    ARG1.layout() == at::kSparseCsr,                                            \
    #OPERATOR_NAME " expects sparse CSR tensor"                                 \
  );                                                                            \
                                                                                \
  at::Tensor values = ARG1.values();                                            \
                                                                                \
  at::Tensor new_values = c10::Dispatcher::singleton()                          \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")                     \
    .typed<at::Tensor(const at::Tensor&)>()                                     \
    .call(values);                                                              \
                                                                                \
  return at::_sparse_csr_tensor_unsafe(                                         \
    ARG1.crow_indices(),                                                        \
    ARG1.col_indices(),                                                         \
    new_values,                                                                 \
    ARG1.sizes(),                                                               \
    ARG1.options().layout(at::kSparseCsr).dtype(new_values.scalar_type())       \
  );                                                                            \
}                                                                               \
                                                                                \
inline at::Tensor OPERATOR_NAME##_backward(                                     \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    gradient_output.layout() == at::kSparseCsr,                                 \
    #OPERATOR_NAME "_backward expects sparse CSR gradient"                      \
  );                                                                            \
  TORCH_CHECK(                                                                  \
    ARG1.layout() == at::kSparseCsr,                                            \
    #OPERATOR_NAME "_backward expects sparse CSR input"                         \
  );                                                                            \
                                                                                \
  at::Tensor grad_values = gradient_output.values();                            \
  at::Tensor input_values = ARG1.values();                                      \
                                                                                \
  at::Tensor new_grad_values = c10::Dispatcher::singleton()                     \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward", "")         \
    .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()                  \
    .call(grad_values, input_values);                                           \
                                                                                \
  return at::_sparse_csr_tensor_unsafe(                                         \
    ARG1.crow_indices(),                                                        \
    ARG1.col_indices(),                                                         \
    new_grad_values,                                                            \
    ARG1.sizes(),                                                               \
    ARG1.options().layout(at::kSparseCsr).dtype(new_grad_values.scalar_type())  \
  );                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> OPERATOR_NAME##_backward_backward(    \
  const at::Tensor& gradient_gradient_##ARG1,                                   \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  if (!gradient_gradient_##ARG1.defined()) {                                    \
    return {};                                                                  \
  }                                                                             \
                                                                                \
  TORCH_CHECK(                                                                  \
    gradient_gradient_##ARG1.layout() == at::kSparseCsr,                        \
    #OPERATOR_NAME "_backward_backward expects sparse CSR grad_grad"            \
  );                                                                            \
  TORCH_CHECK(                                                                  \
    gradient_output.layout() == at::kSparseCsr,                                 \
    #OPERATOR_NAME "_backward_backward expects sparse CSR gradient"             \
  );                                                                            \
  TORCH_CHECK(                                                                  \
    ARG1.layout() == at::kSparseCsr,                                            \
    #OPERATOR_NAME "_backward_backward expects sparse CSR input"                \
  );                                                                            \
                                                                                \
  at::Tensor gg_values = gradient_gradient_##ARG1.values();                     \
  at::Tensor grad_values = gradient_output.values();                            \
  at::Tensor input_values = ARG1.values();                                      \
                                                                                \
  auto [new_gg_out_values, new_grad_values] = c10::Dispatcher::singleton()      \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward_backward", "")\
    .typed<std::tuple<at::Tensor, at::Tensor>(                                  \
      const at::Tensor&, const at::Tensor&, const at::Tensor&                   \
    )>()                                                                        \
    .call(gg_values, grad_values, input_values);                                \
                                                                                \
  at::Tensor grad_grad_output;                                                  \
  at::Tensor grad_##ARG1;                                                       \
                                                                                \
  if (new_gg_out_values.defined()) {                                            \
    grad_grad_output = at::_sparse_csr_tensor_unsafe(                           \
      gradient_output.crow_indices(),                                           \
      gradient_output.col_indices(),                                            \
      new_gg_out_values,                                                        \
      gradient_output.sizes(),                                                  \
      gradient_output.options().layout(at::kSparseCsr)                          \
        .dtype(new_gg_out_values.scalar_type())                                 \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (new_grad_values.defined()) {                                              \
    grad_##ARG1 = at::_sparse_csr_tensor_unsafe(                                \
      ARG1.crow_indices(),                                                      \
      ARG1.col_indices(),                                                       \
      new_grad_values,                                                          \
      ARG1.sizes(),                                                             \
      ARG1.options().layout(at::kSparseCsr)                                     \
        .dtype(new_grad_values.scalar_type())                                   \
    );                                                                          \
  }                                                                             \
                                                                                \
  return {grad_grad_output, grad_##ARG1};                                       \
}                                                                               \
                                                                                \
}  /* namespace torchscience::sparse::csr::cuda::NAMESPACE */                   \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, SparseCsrCUDA, module) {                       \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::sparse::csr::cuda::NAMESPACE::OPERATOR_NAME                  \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::sparse::csr::cuda::NAMESPACE::OPERATOR_NAME##_backward       \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::sparse::csr::cuda::NAMESPACE::OPERATOR_NAME##_backward_backward\
  );                                                                            \
}

#define SPARSE_CSR_CUDA_BINARY_OPERATOR(                                        \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::sparse::csr::cuda::NAMESPACE {                          \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    ARG1.layout() == at::kSparseCsr && ARG2.layout() == at::kSparseCsr,         \
    #OPERATOR_NAME " expects sparse CSR tensors"                                \
  );                                                                            \
                                                                                \
  at::Tensor values1 = ARG1.values();                                           \
  at::Tensor values2 = ARG2.values();                                           \
                                                                                \
  at::Tensor new_values = c10::Dispatcher::singleton()                          \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")                     \
    .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()                  \
    .call(values1, values2);                                                    \
                                                                                \
  return at::_sparse_csr_tensor_unsafe(                                         \
    ARG1.crow_indices(),                                                        \
    ARG1.col_indices(),                                                         \
    new_values,                                                                 \
    ARG1.sizes(),                                                               \
    ARG1.options().layout(at::kSparseCsr).dtype(new_values.scalar_type())       \
  );                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> OPERATOR_NAME##_backward(             \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    gradient_output.layout() == at::kSparseCsr,                                 \
    #OPERATOR_NAME "_backward expects sparse CSR gradient"                      \
  );                                                                            \
  TORCH_CHECK(                                                                  \
    ARG1.layout() == at::kSparseCsr && ARG2.layout() == at::kSparseCsr,         \
    #OPERATOR_NAME "_backward expects sparse CSR inputs"                        \
  );                                                                            \
                                                                                \
  at::Tensor grad_values = gradient_output.values();                            \
  at::Tensor values1 = ARG1.values();                                           \
  at::Tensor values2 = ARG2.values();                                           \
                                                                                \
  auto [new_grad1, new_grad2] = c10::Dispatcher::singleton()                    \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward", "")         \
    .typed<std::tuple<at::Tensor, at::Tensor>(                                  \
      const at::Tensor&, const at::Tensor&, const at::Tensor&                   \
    )>()                                                                        \
    .call(grad_values, values1, values2);                                       \
                                                                                \
  at::Tensor grad_##ARG1;                                                       \
  at::Tensor grad_##ARG2;                                                       \
                                                                                \
  if (new_grad1.defined()) {                                                    \
    grad_##ARG1 = at::_sparse_csr_tensor_unsafe(                                \
      ARG1.crow_indices(),                                                      \
      ARG1.col_indices(),                                                       \
      new_grad1,                                                                \
      ARG1.sizes(),                                                             \
      ARG1.options().layout(at::kSparseCsr).dtype(new_grad1.scalar_type())      \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (new_grad2.defined()) {                                                    \
    grad_##ARG2 = at::_sparse_csr_tensor_unsafe(                                \
      ARG2.crow_indices(),                                                      \
      ARG2.col_indices(),                                                       \
      new_grad2,                                                                \
      ARG2.sizes(),                                                             \
      ARG2.options().layout(at::kSparseCsr).dtype(new_grad2.scalar_type())      \
    );                                                                          \
  }                                                                             \
                                                                                \
  return {grad_##ARG1, grad_##ARG2};                                            \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor>                           \
OPERATOR_NAME##_backward_backward(                                              \
  const at::Tensor& gg_##ARG1,                                                  \
  const at::Tensor& gg_##ARG2,                                                  \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  at::Tensor gg1_values = gg_##ARG1.defined() &&                                \
    gg_##ARG1.layout() == at::kSparseCsr ? gg_##ARG1.values() : gg_##ARG1;      \
  at::Tensor gg2_values = gg_##ARG2.defined() &&                                \
    gg_##ARG2.layout() == at::kSparseCsr ? gg_##ARG2.values() : gg_##ARG2;      \
  at::Tensor grad_values = gradient_output.values();                            \
  at::Tensor values1 = ARG1.values();                                           \
  at::Tensor values2 = ARG2.values();                                           \
                                                                                \
  auto [new_gg_out, new_grad1, new_grad2] = c10::Dispatcher::singleton()        \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward_backward", "")\
    .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(                      \
      const at::Tensor&, const at::Tensor&, const at::Tensor&,                  \
      const at::Tensor&, const at::Tensor&                                      \
    )>()                                                                        \
    .call(gg1_values, gg2_values, grad_values, values1, values2);               \
                                                                                \
  at::Tensor grad_grad_output;                                                  \
  at::Tensor grad_##ARG1;                                                       \
  at::Tensor grad_##ARG2;                                                       \
                                                                                \
  if (new_gg_out.defined()) {                                                   \
    grad_grad_output = at::_sparse_csr_tensor_unsafe(                           \
      gradient_output.crow_indices(),                                           \
      gradient_output.col_indices(),                                            \
      new_gg_out,                                                               \
      gradient_output.sizes(),                                                  \
      gradient_output.options().layout(at::kSparseCsr)                          \
        .dtype(new_gg_out.scalar_type())                                        \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (new_grad1.defined()) {                                                    \
    grad_##ARG1 = at::_sparse_csr_tensor_unsafe(                                \
      ARG1.crow_indices(),                                                      \
      ARG1.col_indices(),                                                       \
      new_grad1,                                                                \
      ARG1.sizes(),                                                             \
      ARG1.options().layout(at::kSparseCsr).dtype(new_grad1.scalar_type())      \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (new_grad2.defined()) {                                                    \
    grad_##ARG2 = at::_sparse_csr_tensor_unsafe(                                \
      ARG2.crow_indices(),                                                      \
      ARG2.col_indices(),                                                       \
      new_grad2,                                                                \
      ARG2.sizes(),                                                             \
      ARG2.options().layout(at::kSparseCsr).dtype(new_grad2.scalar_type())      \
    );                                                                          \
  }                                                                             \
                                                                                \
  return {grad_grad_output, grad_##ARG1, grad_##ARG2};                          \
}                                                                               \
                                                                                \
}  /* namespace torchscience::sparse::csr::cuda::NAMESPACE */                   \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, SparseCsrCUDA, module) {                       \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::sparse::csr::cuda::NAMESPACE::OPERATOR_NAME                  \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::sparse::csr::cuda::NAMESPACE::OPERATOR_NAME##_backward       \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::sparse::csr::cuda::NAMESPACE::OPERATOR_NAME##_backward_backward\
  );                                                                            \
}

#define SPARSE_CSR_CUDA_TERNARY_OPERATOR(                                       \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::sparse::csr::cuda::NAMESPACE {                          \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    ARG1.layout() == at::kSparseCsr && ARG2.layout() == at::kSparseCsr &&       \
    ARG3.layout() == at::kSparseCsr,                                            \
    #OPERATOR_NAME " expects sparse CSR tensors"                                \
  );                                                                            \
                                                                                \
  at::Tensor values1 = ARG1.values();                                           \
  at::Tensor values2 = ARG2.values();                                           \
  at::Tensor values3 = ARG3.values();                                           \
                                                                                \
  at::Tensor new_values = c10::Dispatcher::singleton()                          \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")                     \
    .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()\
    .call(values1, values2, values3);                                           \
                                                                                \
  return at::_sparse_csr_tensor_unsafe(                                         \
    ARG1.crow_indices(), ARG1.col_indices(), new_values, ARG1.sizes(),          \
    ARG1.options().layout(at::kSparseCsr).dtype(new_values.scalar_type())       \
  );                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> OPERATOR_NAME##_backward( \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3                                                        \
) {                                                                             \
  at::Tensor grad_values = gradient_output.values();                            \
  at::Tensor values1 = ARG1.values();                                           \
  at::Tensor values2 = ARG2.values();                                           \
  at::Tensor values3 = ARG3.values();                                           \
                                                                                \
  auto [new_grad1, new_grad2, new_grad3] = c10::Dispatcher::singleton()         \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward", "")         \
    .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(                      \
      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&\
    )>()                                                                        \
    .call(grad_values, values1, values2, values3);                              \
                                                                                \
  at::Tensor grad_##ARG1, grad_##ARG2, grad_##ARG3;                             \
                                                                                \
  if (new_grad1.defined()) {                                                    \
    grad_##ARG1 = at::_sparse_csr_tensor_unsafe(                                \
      ARG1.crow_indices(), ARG1.col_indices(), new_grad1, ARG1.sizes(),         \
      ARG1.options().layout(at::kSparseCsr).dtype(new_grad1.scalar_type())      \
    );                                                                          \
  }                                                                             \
  if (new_grad2.defined()) {                                                    \
    grad_##ARG2 = at::_sparse_csr_tensor_unsafe(                                \
      ARG2.crow_indices(), ARG2.col_indices(), new_grad2, ARG2.sizes(),         \
      ARG2.options().layout(at::kSparseCsr).dtype(new_grad2.scalar_type())      \
    );                                                                          \
  }                                                                             \
  if (new_grad3.defined()) {                                                    \
    grad_##ARG3 = at::_sparse_csr_tensor_unsafe(                                \
      ARG3.crow_indices(), ARG3.col_indices(), new_grad3, ARG3.sizes(),         \
      ARG3.options().layout(at::kSparseCsr).dtype(new_grad3.scalar_type())      \
    );                                                                          \
  }                                                                             \
                                                                                \
  return {grad_##ARG1, grad_##ARG2, grad_##ARG3};                               \
}                                                                               \
                                                                                \
}  /* namespace torchscience::sparse::csr::cuda::NAMESPACE */                   \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, SparseCsrCUDA, module) {                       \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::sparse::csr::cuda::NAMESPACE::OPERATOR_NAME                  \
  );                                                                            \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::sparse::csr::cuda::NAMESPACE::OPERATOR_NAME##_backward       \
  );                                                                            \
}

#define SPARSE_CSR_CUDA_QUATERNARY_OPERATOR(                                    \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3,                                                                         \
  ARG4                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::sparse::csr::cuda::NAMESPACE {                          \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3,                                                       \
  const at::Tensor& ARG4                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    ARG1.layout() == at::kSparseCsr && ARG2.layout() == at::kSparseCsr &&       \
    ARG3.layout() == at::kSparseCsr && ARG4.layout() == at::kSparseCsr,         \
    #OPERATOR_NAME " expects sparse CSR tensors"                                \
  );                                                                            \
                                                                                \
  at::Tensor values1 = ARG1.values();                                           \
  at::Tensor values2 = ARG2.values();                                           \
  at::Tensor values3 = ARG3.values();                                           \
  at::Tensor values4 = ARG4.values();                                           \
                                                                                \
  at::Tensor new_values = c10::Dispatcher::singleton()                          \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")                     \
    .typed<at::Tensor(                                                          \
      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&\
    )>()                                                                        \
    .call(values1, values2, values3, values4);                                  \
                                                                                \
  return at::_sparse_csr_tensor_unsafe(                                         \
    ARG1.crow_indices(), ARG1.col_indices(), new_values, ARG1.sizes(),          \
    ARG1.options().layout(at::kSparseCsr).dtype(new_values.scalar_type())       \
  );                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>               \
OPERATOR_NAME##_backward(                                                       \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3,                                                       \
  const at::Tensor& ARG4                                                        \
) {                                                                             \
  at::Tensor grad_values = gradient_output.values();                            \
  at::Tensor values1 = ARG1.values();                                           \
  at::Tensor values2 = ARG2.values();                                           \
  at::Tensor values3 = ARG3.values();                                           \
  at::Tensor values4 = ARG4.values();                                           \
                                                                                \
  auto [new_grad1, new_grad2, new_grad3, new_grad4] =                           \
    c10::Dispatcher::singleton()                                                \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward", "")         \
    .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(          \
      const at::Tensor&, const at::Tensor&, const at::Tensor&,                  \
      const at::Tensor&, const at::Tensor&                                      \
    )>()                                                                        \
    .call(grad_values, values1, values2, values3, values4);                     \
                                                                                \
  at::Tensor grad_##ARG1, grad_##ARG2, grad_##ARG3, grad_##ARG4;                \
                                                                                \
  if (new_grad1.defined()) {                                                    \
    grad_##ARG1 = at::_sparse_csr_tensor_unsafe(                                \
      ARG1.crow_indices(), ARG1.col_indices(), new_grad1, ARG1.sizes(),         \
      ARG1.options().layout(at::kSparseCsr).dtype(new_grad1.scalar_type())      \
    );                                                                          \
  }                                                                             \
  if (new_grad2.defined()) {                                                    \
    grad_##ARG2 = at::_sparse_csr_tensor_unsafe(                                \
      ARG2.crow_indices(), ARG2.col_indices(), new_grad2, ARG2.sizes(),         \
      ARG2.options().layout(at::kSparseCsr).dtype(new_grad2.scalar_type())      \
    );                                                                          \
  }                                                                             \
  if (new_grad3.defined()) {                                                    \
    grad_##ARG3 = at::_sparse_csr_tensor_unsafe(                                \
      ARG3.crow_indices(), ARG3.col_indices(), new_grad3, ARG3.sizes(),         \
      ARG3.options().layout(at::kSparseCsr).dtype(new_grad3.scalar_type())      \
    );                                                                          \
  }                                                                             \
  if (new_grad4.defined()) {                                                    \
    grad_##ARG4 = at::_sparse_csr_tensor_unsafe(                                \
      ARG4.crow_indices(), ARG4.col_indices(), new_grad4, ARG4.sizes(),         \
      ARG4.options().layout(at::kSparseCsr).dtype(new_grad4.scalar_type())      \
    );                                                                          \
  }                                                                             \
                                                                                \
  return {grad_##ARG1, grad_##ARG2, grad_##ARG3, grad_##ARG4};                  \
}                                                                               \
                                                                                \
}  /* namespace torchscience::sparse::csr::cuda::NAMESPACE */                   \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, SparseCsrCUDA, module) {                       \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::sparse::csr::cuda::NAMESPACE::OPERATOR_NAME                  \
  );                                                                            \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::sparse::csr::cuda::NAMESPACE::OPERATOR_NAME##_backward       \
  );                                                                            \
}
