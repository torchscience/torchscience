#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

#define QUANTIZED_CUDA_UNARY_OPERATOR(                                          \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::quantized::cuda::NAMESPACE {                            \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    ARG1.is_quantized(),                                                        \
    #OPERATOR_NAME " expects quantized tensor"                                  \
  );                                                                            \
                                                                                \
  at::Tensor dequantized = ARG1.dequantize();                                   \
                                                                                \
  at::Tensor result = c10::Dispatcher::singleton()                              \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")                     \
    .typed<at::Tensor(const at::Tensor&)>()                                     \
    .call(dequantized);                                                         \
                                                                                \
  return at::quantize_per_tensor(                                               \
    result,                                                                     \
    ARG1.q_scale(),                                                             \
    ARG1.q_zero_point(),                                                        \
    ARG1.scalar_type()                                                          \
  );                                                                            \
}                                                                               \
                                                                                \
inline at::Tensor OPERATOR_NAME##_backward(                                     \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    gradient_output.is_quantized(),                                             \
    #OPERATOR_NAME "_backward expects quantized gradient"                       \
  );                                                                            \
  TORCH_CHECK(                                                                  \
    ARG1.is_quantized(),                                                        \
    #OPERATOR_NAME "_backward expects quantized input"                          \
  );                                                                            \
                                                                                \
  at::Tensor grad_dequantized = gradient_output.dequantize();                   \
  at::Tensor input_dequantized = ARG1.dequantize();                             \
                                                                                \
  at::Tensor result = c10::Dispatcher::singleton()                              \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward", "")         \
    .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()                  \
    .call(grad_dequantized, input_dequantized);                                 \
                                                                                \
  return at::quantize_per_tensor(                                               \
    result,                                                                     \
    ARG1.q_scale(),                                                             \
    ARG1.q_zero_point(),                                                        \
    ARG1.scalar_type()                                                          \
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
    gradient_gradient_##ARG1.is_quantized(),                                    \
    #OPERATOR_NAME "_backward_backward expects quantized grad_grad"             \
  );                                                                            \
  TORCH_CHECK(                                                                  \
    gradient_output.is_quantized(),                                             \
    #OPERATOR_NAME "_backward_backward expects quantized gradient"              \
  );                                                                            \
  TORCH_CHECK(                                                                  \
    ARG1.is_quantized(),                                                        \
    #OPERATOR_NAME "_backward_backward expects quantized input"                 \
  );                                                                            \
                                                                                \
  at::Tensor gg_dequantized = gradient_gradient_##ARG1.dequantize();            \
  at::Tensor grad_dequantized = gradient_output.dequantize();                   \
  at::Tensor input_dequantized = ARG1.dequantize();                             \
                                                                                \
  auto [grad_grad_out, grad_input] = c10::Dispatcher::singleton()               \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward_backward", "")\
    .typed<std::tuple<at::Tensor, at::Tensor>(                                  \
      const at::Tensor&, const at::Tensor&, const at::Tensor&                   \
    )>()                                                                        \
    .call(gg_dequantized, grad_dequantized, input_dequantized);                 \
                                                                                \
  at::Tensor quantized_grad_grad_out;                                           \
  at::Tensor quantized_grad_input;                                              \
                                                                                \
  if (grad_grad_out.defined()) {                                                \
    quantized_grad_grad_out = at::quantize_per_tensor(                          \
      grad_grad_out,                                                            \
      gradient_output.q_scale(),                                                \
      gradient_output.q_zero_point(),                                           \
      gradient_output.scalar_type()                                             \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (grad_input.defined()) {                                                   \
    quantized_grad_input = at::quantize_per_tensor(                             \
      grad_input,                                                               \
      ARG1.q_scale(),                                                           \
      ARG1.q_zero_point(),                                                      \
      ARG1.scalar_type()                                                        \
    );                                                                          \
  }                                                                             \
                                                                                \
  return {quantized_grad_grad_out, quantized_grad_input};                       \
}                                                                               \
                                                                                \
}  /* namespace torchscience::quantized::cuda::NAMESPACE */                     \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, QuantizedCUDA, module) {                       \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::quantized::cuda::NAMESPACE::OPERATOR_NAME                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::quantized::cuda::NAMESPACE::OPERATOR_NAME##_backward         \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::quantized::cuda::NAMESPACE::OPERATOR_NAME##_backward_backward\
  );                                                                            \
}

#define QUANTIZED_CUDA_BINARY_OPERATOR(                                         \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::quantized::cuda::NAMESPACE {                            \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    ARG1.is_quantized() && ARG2.is_quantized(),                                 \
    #OPERATOR_NAME " expects quantized tensors"                                 \
  );                                                                            \
                                                                                \
  at::Tensor dequant1 = ARG1.dequantize();                                      \
  at::Tensor dequant2 = ARG2.dequantize();                                      \
                                                                                \
  at::Tensor result = c10::Dispatcher::singleton()                              \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")                     \
    .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()                  \
    .call(dequant1, dequant2);                                                  \
                                                                                \
  return at::quantize_per_tensor(                                               \
    result,                                                                     \
    ARG1.q_scale(),                                                             \
    ARG1.q_zero_point(),                                                        \
    ARG1.scalar_type()                                                          \
  );                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> OPERATOR_NAME##_backward(             \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    gradient_output.is_quantized(),                                             \
    #OPERATOR_NAME "_backward expects quantized gradient"                       \
  );                                                                            \
  TORCH_CHECK(                                                                  \
    ARG1.is_quantized() && ARG2.is_quantized(),                                 \
    #OPERATOR_NAME "_backward expects quantized inputs"                         \
  );                                                                            \
                                                                                \
  at::Tensor grad_dequant = gradient_output.dequantize();                       \
  at::Tensor dequant1 = ARG1.dequantize();                                      \
  at::Tensor dequant2 = ARG2.dequantize();                                      \
                                                                                \
  auto [grad1, grad2] = c10::Dispatcher::singleton()                            \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward", "")         \
    .typed<std::tuple<at::Tensor, at::Tensor>(                                  \
      const at::Tensor&, const at::Tensor&, const at::Tensor&                   \
    )>()                                                                        \
    .call(grad_dequant, dequant1, dequant2);                                    \
                                                                                \
  at::Tensor quant_grad1;                                                       \
  at::Tensor quant_grad2;                                                       \
                                                                                \
  if (grad1.defined()) {                                                        \
    quant_grad1 = at::quantize_per_tensor(                                      \
      grad1, ARG1.q_scale(), ARG1.q_zero_point(), ARG1.scalar_type()            \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (grad2.defined()) {                                                        \
    quant_grad2 = at::quantize_per_tensor(                                      \
      grad2, ARG2.q_scale(), ARG2.q_zero_point(), ARG2.scalar_type()            \
    );                                                                          \
  }                                                                             \
                                                                                \
  return {quant_grad1, quant_grad2};                                            \
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
  at::Tensor gg1_dequant = gg_##ARG1.defined() && gg_##ARG1.is_quantized()      \
    ? gg_##ARG1.dequantize() : gg_##ARG1;                                       \
  at::Tensor gg2_dequant = gg_##ARG2.defined() && gg_##ARG2.is_quantized()      \
    ? gg_##ARG2.dequantize() : gg_##ARG2;                                       \
  at::Tensor grad_dequant = gradient_output.dequantize();                       \
  at::Tensor dequant1 = ARG1.dequantize();                                      \
  at::Tensor dequant2 = ARG2.dequantize();                                      \
                                                                                \
  auto [gg_out, grad1, grad2] = c10::Dispatcher::singleton()                    \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward_backward", "")\
    .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(                      \
      const at::Tensor&, const at::Tensor&, const at::Tensor&,                  \
      const at::Tensor&, const at::Tensor&                                      \
    )>()                                                                        \
    .call(gg1_dequant, gg2_dequant, grad_dequant, dequant1, dequant2);          \
                                                                                \
  at::Tensor quant_gg_out;                                                      \
  at::Tensor quant_grad1;                                                       \
  at::Tensor quant_grad2;                                                       \
                                                                                \
  if (gg_out.defined()) {                                                       \
    quant_gg_out = at::quantize_per_tensor(                                     \
      gg_out,                                                                   \
      gradient_output.q_scale(),                                                \
      gradient_output.q_zero_point(),                                           \
      gradient_output.scalar_type()                                             \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (grad1.defined()) {                                                        \
    quant_grad1 = at::quantize_per_tensor(                                      \
      grad1, ARG1.q_scale(), ARG1.q_zero_point(), ARG1.scalar_type()            \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (grad2.defined()) {                                                        \
    quant_grad2 = at::quantize_per_tensor(                                      \
      grad2, ARG2.q_scale(), ARG2.q_zero_point(), ARG2.scalar_type()            \
    );                                                                          \
  }                                                                             \
                                                                                \
  return {quant_gg_out, quant_grad1, quant_grad2};                              \
}                                                                               \
                                                                                \
}  /* namespace torchscience::quantized::cuda::NAMESPACE */                     \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, QuantizedCUDA, module) {                       \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::quantized::cuda::NAMESPACE::OPERATOR_NAME                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::quantized::cuda::NAMESPACE::OPERATOR_NAME##_backward         \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::quantized::cuda::NAMESPACE::OPERATOR_NAME##_backward_backward\
  );                                                                            \
}

#define QUANTIZED_CUDA_TERNARY_OPERATOR(                                        \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::quantized::cuda::NAMESPACE {                            \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    ARG1.is_quantized() && ARG2.is_quantized() && ARG3.is_quantized(),          \
    #OPERATOR_NAME " expects quantized tensors"                                 \
  );                                                                            \
                                                                                \
  at::Tensor dequant1 = ARG1.dequantize();                                      \
  at::Tensor dequant2 = ARG2.dequantize();                                      \
  at::Tensor dequant3 = ARG3.dequantize();                                      \
                                                                                \
  at::Tensor result = c10::Dispatcher::singleton()                              \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")                     \
    .typed<at::Tensor(                                                          \
      const at::Tensor&, const at::Tensor&, const at::Tensor&                   \
    )>()                                                                        \
    .call(dequant1, dequant2, dequant3);                                        \
                                                                                \
  return at::quantize_per_tensor(                                               \
    result,                                                                     \
    ARG1.q_scale(),                                                             \
    ARG1.q_zero_point(),                                                        \
    ARG1.scalar_type()                                                          \
  );                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> OPERATOR_NAME##_backward( \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    gradient_output.is_quantized(),                                             \
    #OPERATOR_NAME "_backward expects quantized gradient"                       \
  );                                                                            \
  TORCH_CHECK(                                                                  \
    ARG1.is_quantized() && ARG2.is_quantized() && ARG3.is_quantized(),          \
    #OPERATOR_NAME "_backward expects quantized inputs"                         \
  );                                                                            \
                                                                                \
  at::Tensor grad_dequant = gradient_output.dequantize();                       \
  at::Tensor dequant1 = ARG1.dequantize();                                      \
  at::Tensor dequant2 = ARG2.dequantize();                                      \
  at::Tensor dequant3 = ARG3.dequantize();                                      \
                                                                                \
  auto [grad1, grad2, grad3] = c10::Dispatcher::singleton()                     \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward", "")         \
    .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(                      \
      const at::Tensor&, const at::Tensor&, const at::Tensor&,                  \
      const at::Tensor&                                                         \
    )>()                                                                        \
    .call(grad_dequant, dequant1, dequant2, dequant3);                          \
                                                                                \
  at::Tensor quant_grad1;                                                       \
  at::Tensor quant_grad2;                                                       \
  at::Tensor quant_grad3;                                                       \
                                                                                \
  if (grad1.defined()) {                                                        \
    quant_grad1 = at::quantize_per_tensor(                                      \
      grad1, ARG1.q_scale(), ARG1.q_zero_point(), ARG1.scalar_type()            \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (grad2.defined()) {                                                        \
    quant_grad2 = at::quantize_per_tensor(                                      \
      grad2, ARG2.q_scale(), ARG2.q_zero_point(), ARG2.scalar_type()            \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (grad3.defined()) {                                                        \
    quant_grad3 = at::quantize_per_tensor(                                      \
      grad3, ARG3.q_scale(), ARG3.q_zero_point(), ARG3.scalar_type()            \
    );                                                                          \
  }                                                                             \
                                                                                \
  return {quant_grad1, quant_grad2, quant_grad3};                               \
}                                                                               \
                                                                                \
}  /* namespace torchscience::quantized::cuda::NAMESPACE */                     \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, QuantizedCUDA, module) {                       \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::quantized::cuda::NAMESPACE::OPERATOR_NAME                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::quantized::cuda::NAMESPACE::OPERATOR_NAME##_backward         \
  );                                                                            \
}

#define QUANTIZED_CUDA_QUATERNARY_OPERATOR(                                     \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3,                                                                         \
  ARG4                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::quantized::cuda::NAMESPACE {                            \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3,                                                       \
  const at::Tensor& ARG4                                                        \
) {                                                                             \
  TORCH_CHECK(                                                                  \
    ARG1.is_quantized() && ARG2.is_quantized() &&                               \
    ARG3.is_quantized() && ARG4.is_quantized(),                                 \
    #OPERATOR_NAME " expects quantized tensors"                                 \
  );                                                                            \
                                                                                \
  at::Tensor dequant1 = ARG1.dequantize();                                      \
  at::Tensor dequant2 = ARG2.dequantize();                                      \
  at::Tensor dequant3 = ARG3.dequantize();                                      \
  at::Tensor dequant4 = ARG4.dequantize();                                      \
                                                                                \
  at::Tensor result = c10::Dispatcher::singleton()                              \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")                     \
    .typed<at::Tensor(                                                          \
      const at::Tensor&, const at::Tensor&, const at::Tensor&,                  \
      const at::Tensor&                                                         \
    )>()                                                                        \
    .call(dequant1, dequant2, dequant3, dequant4);                              \
                                                                                \
  return at::quantize_per_tensor(                                               \
    result,                                                                     \
    ARG1.q_scale(),                                                             \
    ARG1.q_zero_point(),                                                        \
    ARG1.scalar_type()                                                          \
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
  TORCH_CHECK(                                                                  \
    gradient_output.is_quantized(),                                             \
    #OPERATOR_NAME "_backward expects quantized gradient"                       \
  );                                                                            \
  TORCH_CHECK(                                                                  \
    ARG1.is_quantized() && ARG2.is_quantized() &&                               \
    ARG3.is_quantized() && ARG4.is_quantized(),                                 \
    #OPERATOR_NAME "_backward expects quantized inputs"                         \
  );                                                                            \
                                                                                \
  at::Tensor grad_dequant = gradient_output.dequantize();                       \
  at::Tensor dequant1 = ARG1.dequantize();                                      \
  at::Tensor dequant2 = ARG2.dequantize();                                      \
  at::Tensor dequant3 = ARG3.dequantize();                                      \
  at::Tensor dequant4 = ARG4.dequantize();                                      \
                                                                                \
  auto [grad1, grad2, grad3, grad4] = c10::Dispatcher::singleton()              \
    .findSchemaOrThrow("torchscience::" #OPERATOR_NAME "_backward", "")         \
    .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(          \
      const at::Tensor&, const at::Tensor&, const at::Tensor&,                  \
      const at::Tensor&, const at::Tensor&                                      \
    )>()                                                                        \
    .call(grad_dequant, dequant1, dequant2, dequant3, dequant4);                \
                                                                                \
  at::Tensor quant_grad1;                                                       \
  at::Tensor quant_grad2;                                                       \
  at::Tensor quant_grad3;                                                       \
  at::Tensor quant_grad4;                                                       \
                                                                                \
  if (grad1.defined()) {                                                        \
    quant_grad1 = at::quantize_per_tensor(                                      \
      grad1, ARG1.q_scale(), ARG1.q_zero_point(), ARG1.scalar_type()            \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (grad2.defined()) {                                                        \
    quant_grad2 = at::quantize_per_tensor(                                      \
      grad2, ARG2.q_scale(), ARG2.q_zero_point(), ARG2.scalar_type()            \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (grad3.defined()) {                                                        \
    quant_grad3 = at::quantize_per_tensor(                                      \
      grad3, ARG3.q_scale(), ARG3.q_zero_point(), ARG3.scalar_type()            \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (grad4.defined()) {                                                        \
    quant_grad4 = at::quantize_per_tensor(                                      \
      grad4, ARG4.q_scale(), ARG4.q_zero_point(), ARG4.scalar_type()            \
    );                                                                          \
  }                                                                             \
                                                                                \
  return {quant_grad1, quant_grad2, quant_grad3, quant_grad4};                  \
}                                                                               \
                                                                                \
}  /* namespace torchscience::quantized::cuda::NAMESPACE */                     \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, QuantizedCUDA, module) {                       \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::quantized::cuda::NAMESPACE::OPERATOR_NAME                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::quantized::cuda::NAMESPACE::OPERATOR_NAME##_backward         \
  );                                                                            \
}
