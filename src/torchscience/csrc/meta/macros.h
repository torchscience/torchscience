#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

#define META_UNARY_OPERATOR(                                                    \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::meta::NAMESPACE {                                       \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  const at::Tensor output;                                                      \
                                                                                \
  return at::TensorIteratorConfig()                                             \
    .add_output(output)                                                         \
    .add_const_input(ARG1)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build()                                                                    \
    .output();                                                                  \
}                                                                               \
                                                                                \
inline at::Tensor OPERATOR_NAME##_backward(                                     \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  const at::Tensor gradient_##ARG1;                                             \
                                                                                \
  return at::TensorIteratorConfig()                                             \
    .add_output(gradient_##ARG1)                                                \
    .add_const_input(gradient_output)                                           \
    .add_const_input(ARG1)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build()                                                                    \
    .output();                                                                  \
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
  const at::Tensor gradient_gradient_output;                                    \
  const at::Tensor gradient_##ARG1;                                             \
                                                                                \
  const auto iterator = at::TensorIteratorConfig()                              \
    .add_output(gradient_gradient_output)                                       \
    .add_output(gradient_##ARG1)                                                \
    .add_const_input(gradient_gradient_##ARG1)                                  \
    .add_const_input(gradient_output)                                           \
    .add_const_input(ARG1)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  return {                                                                      \
    iterator.output(0),                                                         \
    iterator.output(1)                                                          \
  };                                                                            \
}                                                                               \
                                                                                \
}  /* namespace torchscience::meta::NAMESPACE */                                \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                                \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME                               \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME##_backward                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME##_backward_backward           \
  );                                                                            \
}

#define META_BINARY_OPERATOR(                                                   \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::meta::NAMESPACE {                                       \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  const at::Tensor output;                                                      \
                                                                                \
  return at::TensorIteratorConfig()                                             \
    .add_output(output)                                                         \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build()                                                                    \
    .output();                                                                  \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> OPERATOR_NAME##_backward(             \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  const at::Tensor gradient_##ARG1;                                             \
  const at::Tensor gradient_##ARG2;                                             \
                                                                                \
  const auto iterator = at::TensorIteratorConfig()                              \
    .add_output(gradient_##ARG1)                                                \
    .add_output(gradient_##ARG2)                                                \
    .add_const_input(gradient_output)                                           \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  return {                                                                      \
    iterator.output(0),                                                         \
    iterator.output(1)                                                          \
  };                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor>                           \
OPERATOR_NAME##_backward_backward(                                              \
  const at::Tensor& gradient_gradient_##ARG1,                                   \
  const at::Tensor& gradient_gradient_##ARG2,                                   \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  const bool has_gradient_gradient_##ARG1 =                                     \
    gradient_gradient_##ARG1.defined();                                         \
  const bool has_gradient_gradient_##ARG2 =                                     \
    gradient_gradient_##ARG2.defined();                                         \
                                                                                \
  if (!has_gradient_gradient_##ARG1 && !has_gradient_gradient_##ARG2) {         \
    return {};                                                                  \
  }                                                                             \
                                                                                \
  const at::Tensor gradient_gradient_output;                                    \
  const at::Tensor gradient_##ARG1;                                             \
  const at::Tensor gradient_##ARG2;                                             \
                                                                                \
  at::Tensor gradient_gradient_##ARG1##_input;                                  \
                                                                                \
  if (has_gradient_gradient_##ARG1) {                                           \
    gradient_gradient_##ARG1##_input = gradient_gradient_##ARG1;                \
  } else {                                                                      \
    gradient_gradient_##ARG1##_input = zeros_like(gradient_output);             \
  }                                                                             \
                                                                                \
  at::Tensor gradient_gradient_##ARG2##_input;                                  \
                                                                                \
  if (has_gradient_gradient_##ARG2) {                                           \
    gradient_gradient_##ARG2##_input = gradient_gradient_##ARG2;                \
  } else {                                                                      \
    gradient_gradient_##ARG2##_input = zeros_like(gradient_output);             \
  }                                                                             \
                                                                                \
  const auto iterator = at::TensorIteratorConfig()                              \
    .add_output(gradient_gradient_output)                                       \
    .add_output(gradient_##ARG1)                                                \
    .add_output(gradient_##ARG2)                                                \
    .add_const_input(gradient_gradient_##ARG1##_input)                          \
    .add_const_input(gradient_gradient_##ARG2##_input)                          \
    .add_const_input(gradient_output)                                           \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  return {                                                                      \
    iterator.output(0),                                                         \
    iterator.output(1),                                                         \
    iterator.output(2)                                                          \
  };                                                                            \
}                                                                               \
                                                                                \
}  /* namespace torchscience::meta::NAMESPACE */                                \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                                \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME                               \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME##_backward                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME##_backward_backward           \
  );                                                                            \
}

#define META_TERNARY_OPERATOR(                                                  \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::meta::NAMESPACE {                                       \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3                                                        \
) {                                                                             \
  const at::Tensor output;                                                      \
                                                                                \
  return at::TensorIteratorConfig()                                             \
    .add_output(output)                                                         \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .add_const_input(ARG3)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build()                                                                    \
    .output();                                                                  \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor>                           \
OPERATOR_NAME##_backward(                                                       \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3                                                        \
) {                                                                             \
  const at::Tensor gradient_##ARG1;                                             \
  const at::Tensor gradient_##ARG2;                                             \
  const at::Tensor gradient_##ARG3;                                             \
                                                                                \
  const auto iterator = at::TensorIteratorConfig()                              \
    .add_output(gradient_##ARG1)                                                \
    .add_output(gradient_##ARG2)                                                \
    .add_output(gradient_##ARG3)                                                \
    .add_const_input(gradient_output)                                           \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .add_const_input(ARG3)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  return {                                                                      \
    iterator.output(0),                                                         \
    iterator.output(1),                                                         \
    iterator.output(2)                                                          \
  };                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>               \
OPERATOR_NAME##_backward_backward(                                              \
  const at::Tensor& gradient_gradient_##ARG1,                                   \
  const at::Tensor& gradient_gradient_##ARG2,                                   \
  const at::Tensor& gradient_gradient_##ARG3,                                   \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3                                                        \
) {                                                                             \
  const bool has_gradient_gradient_##ARG1 =                                     \
    gradient_gradient_##ARG1.defined();                                         \
  const bool has_gradient_gradient_##ARG2 =                                     \
    gradient_gradient_##ARG2.defined();                                         \
  const bool has_gradient_gradient_##ARG3 =                                     \
    gradient_gradient_##ARG3.defined();                                         \
                                                                                \
  if (!has_gradient_gradient_##ARG1 &&                                          \
      !has_gradient_gradient_##ARG2 &&                                          \
      !has_gradient_gradient_##ARG3) {                                          \
    return {};                                                                  \
  }                                                                             \
                                                                                \
  const at::Tensor gradient_gradient_output;                                    \
  const at::Tensor gradient_##ARG1;                                             \
  const at::Tensor gradient_##ARG2;                                             \
  const at::Tensor gradient_##ARG3;                                             \
                                                                                \
  at::Tensor gradient_gradient_##ARG1##_input;                                  \
                                                                                \
  if (has_gradient_gradient_##ARG1) {                                           \
    gradient_gradient_##ARG1##_input = gradient_gradient_##ARG1;                \
  } else {                                                                      \
    gradient_gradient_##ARG1##_input = zeros_like(gradient_output);             \
  }                                                                             \
                                                                                \
  at::Tensor gradient_gradient_##ARG2##_input;                                  \
                                                                                \
  if (has_gradient_gradient_##ARG2) {                                           \
    gradient_gradient_##ARG2##_input = gradient_gradient_##ARG2;                \
  } else {                                                                      \
    gradient_gradient_##ARG2##_input = zeros_like(gradient_output);             \
  }                                                                             \
                                                                                \
  at::Tensor gradient_gradient_##ARG3##_input;                                  \
                                                                                \
  if (has_gradient_gradient_##ARG3) {                                           \
    gradient_gradient_##ARG3##_input = gradient_gradient_##ARG3;                \
  } else {                                                                      \
    gradient_gradient_##ARG3##_input = zeros_like(gradient_output);             \
  }                                                                             \
                                                                                \
  const auto iterator = at::TensorIteratorConfig()                              \
    .add_output(gradient_gradient_output)                                       \
    .add_output(gradient_##ARG1)                                                \
    .add_output(gradient_##ARG2)                                                \
    .add_output(gradient_##ARG3)                                                \
    .add_const_input(gradient_gradient_##ARG1##_input)                          \
    .add_const_input(gradient_gradient_##ARG2##_input)                          \
    .add_const_input(gradient_gradient_##ARG3##_input)                          \
    .add_const_input(gradient_output)                                           \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .add_const_input(ARG3)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  return {                                                                      \
    iterator.output(0),                                                         \
    iterator.output(1),                                                         \
    iterator.output(2),                                                         \
    iterator.output(3)                                                          \
  };                                                                            \
}                                                                               \
                                                                                \
}  /* namespace torchscience::meta::NAMESPACE */                                \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                                \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME                               \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME##_backward                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME##_backward_backward           \
  );                                                                            \
}

#define META_QUATERNARY_OPERATOR(                                               \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3,                                                                         \
  ARG4                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::meta::NAMESPACE {                                       \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3,                                                       \
  const at::Tensor& ARG4                                                        \
) {                                                                             \
  const at::Tensor output;                                                      \
                                                                                \
  return at::TensorIteratorConfig()                                             \
    .add_output(output)                                                         \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .add_const_input(ARG3)                                                      \
    .add_const_input(ARG4)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build()                                                                    \
    .output();                                                                  \
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
  const at::Tensor gradient_##ARG1;                                             \
  const at::Tensor gradient_##ARG2;                                             \
  const at::Tensor gradient_##ARG3;                                             \
  const at::Tensor gradient_##ARG4;                                             \
                                                                                \
  const auto iterator = at::TensorIteratorConfig()                              \
    .add_output(gradient_##ARG1)                                                \
    .add_output(gradient_##ARG2)                                                \
    .add_output(gradient_##ARG3)                                                \
    .add_output(gradient_##ARG4)                                                \
    .add_const_input(gradient_output)                                           \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .add_const_input(ARG3)                                                      \
    .add_const_input(ARG4)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  return {                                                                      \
    iterator.output(0),                                                         \
    iterator.output(1),                                                         \
    iterator.output(2),                                                         \
    iterator.output(3)                                                          \
  };                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>   \
OPERATOR_NAME##_backward_backward(                                              \
  const at::Tensor& gradient_gradient_##ARG1,                                   \
  const at::Tensor& gradient_gradient_##ARG2,                                   \
  const at::Tensor& gradient_gradient_##ARG3,                                   \
  const at::Tensor& gradient_gradient_##ARG4,                                   \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3,                                                       \
  const at::Tensor& ARG4                                                        \
) {                                                                             \
  const bool has_gradient_gradient_##ARG1 =                                     \
    gradient_gradient_##ARG1.defined();                                         \
  const bool has_gradient_gradient_##ARG2 =                                     \
    gradient_gradient_##ARG2.defined();                                         \
  const bool has_gradient_gradient_##ARG3 =                                     \
    gradient_gradient_##ARG3.defined();                                         \
  const bool has_gradient_gradient_##ARG4 =                                     \
    gradient_gradient_##ARG4.defined();                                         \
                                                                                \
  if (!has_gradient_gradient_##ARG1 &&                                          \
      !has_gradient_gradient_##ARG2 &&                                          \
      !has_gradient_gradient_##ARG3 &&                                          \
      !has_gradient_gradient_##ARG4) {                                          \
    return {};                                                                  \
  }                                                                             \
                                                                                \
  const at::Tensor gradient_gradient_output;                                    \
  const at::Tensor gradient_##ARG1;                                             \
  const at::Tensor gradient_##ARG2;                                             \
  const at::Tensor gradient_##ARG3;                                             \
  const at::Tensor gradient_##ARG4;                                             \
                                                                                \
  at::Tensor gradient_gradient_##ARG1##_input;                                  \
                                                                                \
  if (has_gradient_gradient_##ARG1) {                                           \
    gradient_gradient_##ARG1##_input = gradient_gradient_##ARG1;                \
  } else {                                                                      \
    gradient_gradient_##ARG1##_input = zeros_like(gradient_output);             \
  }                                                                             \
                                                                                \
  at::Tensor gradient_gradient_##ARG2##_input;                                  \
                                                                                \
  if (has_gradient_gradient_##ARG2) {                                           \
    gradient_gradient_##ARG2##_input = gradient_gradient_##ARG2;                \
  } else {                                                                      \
    gradient_gradient_##ARG2##_input = zeros_like(gradient_output);             \
  }                                                                             \
                                                                                \
  at::Tensor gradient_gradient_##ARG3##_input;                                  \
                                                                                \
  if (has_gradient_gradient_##ARG3) {                                           \
    gradient_gradient_##ARG3##_input = gradient_gradient_##ARG3;                \
  } else {                                                                      \
    gradient_gradient_##ARG3##_input = zeros_like(gradient_output);             \
  }                                                                             \
                                                                                \
  at::Tensor gradient_gradient_##ARG4##_input;                                  \
                                                                                \
  if (has_gradient_gradient_##ARG4) {                                           \
    gradient_gradient_##ARG4##_input = gradient_gradient_##ARG4;                \
  } else {                                                                      \
    gradient_gradient_##ARG4##_input = zeros_like(gradient_output);             \
  }                                                                             \
                                                                                \
  const auto iterator = at::TensorIteratorConfig()                              \
    .add_output(gradient_gradient_output)                                       \
    .add_output(gradient_##ARG1)                                                \
    .add_output(gradient_##ARG2)                                                \
    .add_output(gradient_##ARG3)                                                \
    .add_output(gradient_##ARG4)                                                \
    .add_const_input(gradient_gradient_##ARG1##_input)                          \
    .add_const_input(gradient_gradient_##ARG2##_input)                          \
    .add_const_input(gradient_gradient_##ARG3##_input)                          \
    .add_const_input(gradient_gradient_##ARG4##_input)                          \
    .add_const_input(gradient_output)                                           \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .add_const_input(ARG3)                                                      \
    .add_const_input(ARG4)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  return {                                                                      \
    iterator.output(0),                                                         \
    iterator.output(1),                                                         \
    iterator.output(2),                                                         \
    iterator.output(3),                                                         \
    iterator.output(4)                                                          \
  };                                                                            \
}                                                                               \
                                                                                \
}  /* namespace torchscience::meta::NAMESPACE */                                \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                                \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME                               \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME##_backward                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::meta::NAMESPACE::OPERATOR_NAME##_backward_backward           \
  );                                                                            \
}
