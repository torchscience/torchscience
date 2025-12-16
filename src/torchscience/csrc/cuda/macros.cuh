#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

#define CUDA_UNARY_OPERATOR(                                                    \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::cuda::NAMESPACE {                                       \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  at::Tensor output;                                                            \
                                                                                \
  auto iterator = at::TensorIteratorConfig()                                    \
    .add_output(output)                                                         \
    .add_const_input(ARG1)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_cuda",                                                     \
    [&]() {                                                                     \
      at::native::gpu_kernel(                                                   \
        iterator,                                                               \
        []GPU_LAMBDA(                                                           \
          scalar_t ARG1                                                         \
        ) -> scalar_t {                                                         \
          return impl::NAMESPACE::OPERATOR_NAME<scalar_t>(                      \
            ARG1                                                                \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return iterator.output();                                                     \
}                                                                               \
                                                                                \
inline at::Tensor OPERATOR_NAME##_backward(                                     \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  at::Tensor gradient_##ARG1;                                                   \
                                                                                \
  auto iterator = at::TensorIteratorConfig()                                    \
    .add_output(gradient_##ARG1)                                                \
    .add_const_input(gradient_output)                                           \
    .add_const_input(ARG1)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_backward_cuda",                                            \
    [&]() {                                                                     \
      at::native::gpu_kernel(                                                   \
        iterator,                                                               \
        []GPU_LAMBDA(                                                           \
          scalar_t gradient_output,                                             \
          scalar_t ARG1                                                         \
        ) -> scalar_t {                                                         \
          return impl::NAMESPACE::OPERATOR_NAME##_backward<scalar_t>(           \
            gradient_output,                                                    \
            ARG1                                                                \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return iterator.output();                                                     \
}                                                                               \
                                                                                \
inline std::tuple<                                                              \
  at::Tensor,                                                                   \
  at::Tensor                                                                    \
> OPERATOR_NAME##_backward_backward(                                            \
  const at::Tensor& gradient_gradient_##ARG1,                                   \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  const bool has_gradient_gradient_##ARG1 =                                     \
    gradient_gradient_##ARG1.defined();                                         \
                                                                                \
  if (!has_gradient_gradient_##ARG1) {                                          \
    return std::make_tuple(                                                     \
      at::Tensor(),                                                             \
      at::Tensor()                                                              \
    );                                                                          \
  }                                                                             \
                                                                                \
  at::Tensor gradient_gradient_output;                                          \
  at::Tensor gradient_##ARG1;                                                   \
                                                                                \
  at::Tensor gradient_gradient_##ARG1##_input;                                  \
                                                                                \
  if (has_gradient_gradient_##ARG1) {                                           \
    gradient_gradient_##ARG1##_input = gradient_gradient_##ARG1;                \
  } else {                                                                      \
    gradient_gradient_##ARG1##_input = zeros_like(gradient_output);             \
  }                                                                             \
                                                                                \
  auto iterator = at::TensorIteratorConfig()                                    \
    .add_output(gradient_gradient_output)                                       \
    .add_output(gradient_##ARG1)                                                \
    .add_const_input(gradient_gradient_##ARG1##_input)                          \
    .add_const_input(gradient_output)                                           \
    .add_const_input(ARG1)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_backward_backward_cuda",                                   \
    [&]() {                                                                     \
      at::native::gpu_kernel_multiple_outputs(                                  \
        iterator,                                                               \
        [                                                                       \
          has_gradient_gradient_##ARG1                                          \
        ]GPU_LAMBDA(                                                            \
          scalar_t gradient_gradient_##ARG1,                                    \
          scalar_t gradient_output,                                             \
          scalar_t ARG1                                                         \
        ) -> std::tuple<                                                     \
          scalar_t,                                                             \
          scalar_t                                                              \
        > {                                                                     \
          auto result =                                                         \
            impl::NAMESPACE::OPERATOR_NAME##_backward_backward<scalar_t>(       \
              gradient_gradient_##ARG1,                                         \
              gradient_output,                                                  \
              ARG1,                                                             \
              has_gradient_gradient_##ARG1                                      \
            );                                                                  \
          return std::make_tuple(                                            \
            std::get<0>(result),                                                \
            std::get<1>(result)                                                 \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return std::make_tuple(                                                       \
    iterator.output(0),                                                         \
    iterator.output(1)                                                          \
  );                                                                            \
}                                                                               \
                                                                                \
}  /* namespace torchscience::cuda::NAMESPACE */                                \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {                                \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME                               \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME##_backward                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME##_backward_backward           \
  );                                                                            \
}

#define CUDA_BINARY_OPERATOR(                                                   \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::cuda::NAMESPACE {                                       \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  at::Tensor output;                                                            \
                                                                                \
  auto iterator = at::TensorIteratorConfig()                                    \
    .add_output(output)                                                         \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_cuda",                                                     \
    [&]() {                                                                     \
      at::native::gpu_kernel(                                                   \
        iterator,                                                               \
        []GPU_LAMBDA(                                                           \
          scalar_t ARG1,                                                        \
          scalar_t ARG2                                                         \
        ) -> scalar_t {                                                         \
          return impl::NAMESPACE::OPERATOR_NAME<scalar_t>(                      \
            ARG1,                                                               \
            ARG2                                                                \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return iterator.output();                                                     \
}                                                                               \
                                                                                \
inline std::tuple<                                                              \
  at::Tensor,                                                                   \
  at::Tensor                                                                    \
> OPERATOR_NAME##_backward(                                                     \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  at::Tensor gradient_##ARG1;                                                   \
  at::Tensor gradient_##ARG2;                                                   \
                                                                                \
  auto iterator = at::TensorIteratorConfig()                                    \
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
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_backward_cuda",                                            \
    [&]() {                                                                     \
      at::native::gpu_kernel_multiple_outputs(                                  \
        iterator,                                                               \
        []GPU_LAMBDA(                                                           \
          scalar_t gradient_output,                                             \
          scalar_t ARG1,                                                        \
          scalar_t ARG2                                                         \
        ) -> std::tuple<                                                     \
          scalar_t,                                                             \
          scalar_t                                                              \
        > {                                                                     \
          auto result =                                                         \
            impl::NAMESPACE::OPERATOR_NAME##_backward<scalar_t>(                \
              gradient_output,                                                  \
              ARG1,                                                             \
              ARG2                                                              \
            );                                                                  \
          return std::make_tuple(                                            \
            std::get<0>(result),                                                \
            std::get<1>(result)                                                 \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return std::make_tuple(                                                       \
    iterator.output(0),                                                         \
    iterator.output(1)                                                          \
  );                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<                                                              \
  at::Tensor,                                                                   \
  at::Tensor,                                                                   \
  at::Tensor                                                                    \
> OPERATOR_NAME##_backward_backward(                                            \
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
    return std::make_tuple(                                                     \
      at::Tensor(),                                                             \
      at::Tensor(),                                                             \
      at::Tensor()                                                              \
    );                                                                          \
  }                                                                             \
                                                                                \
  at::Tensor gradient_gradient_output;                                          \
  at::Tensor gradient_##ARG1;                                                   \
  at::Tensor gradient_##ARG2;                                                   \
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
  auto iterator = at::TensorIteratorConfig()                                    \
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
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_backward_backward_cuda",                                   \
    [&]() {                                                                     \
      at::native::gpu_kernel_multiple_outputs(                                  \
        iterator,                                                               \
        [                                                                       \
          has_gradient_gradient_##ARG1,                                         \
          has_gradient_gradient_##ARG2                                          \
        ]GPU_LAMBDA(                                                            \
          scalar_t gradient_gradient_##ARG1,                                    \
          scalar_t gradient_gradient_##ARG2,                                    \
          scalar_t gradient_output,                                             \
          scalar_t ARG1,                                                        \
          scalar_t ARG2                                                         \
        ) -> std::tuple<                                                     \
          scalar_t,                                                             \
          scalar_t,                                                             \
          scalar_t                                                              \
        > {                                                                     \
          auto result =                                                         \
            impl::NAMESPACE::OPERATOR_NAME##_backward_backward<scalar_t>(       \
              gradient_gradient_##ARG1,                                         \
              gradient_gradient_##ARG2,                                         \
              gradient_output,                                                  \
              ARG1,                                                             \
              ARG2,                                                             \
              has_gradient_gradient_##ARG1,                                     \
              has_gradient_gradient_##ARG2                                      \
            );                                                                  \
          return std::make_tuple(                                            \
            std::get<0>(result),                                                \
            std::get<1>(result),                                                \
            std::get<2>(result)                                                 \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return std::make_tuple(                                                       \
    iterator.output(0),                                                         \
    iterator.output(1),                                                         \
    iterator.output(2)                                                          \
  );                                                                            \
}                                                                               \
                                                                                \
}  /* namespace torchscience::cuda::NAMESPACE */                                \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {                                \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME                               \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME##_backward                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME##_backward_backward           \
  );                                                                            \
}

#define CUDA_TERNARY_OPERATOR(                                                  \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::cuda::NAMESPACE {                                       \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3                                                        \
) {                                                                             \
  at::Tensor output;                                                            \
                                                                                \
  auto iterator = at::TensorIteratorConfig()                                    \
    .add_output(output)                                                         \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .add_const_input(ARG3)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_cuda",                                                     \
    [&]() {                                                                     \
      at::native::gpu_kernel(                                                   \
        iterator,                                                               \
        []GPU_LAMBDA(                                                           \
          scalar_t ARG1,                                                        \
          scalar_t ARG2,                                                        \
          scalar_t ARG3                                                         \
        ) -> scalar_t {                                                         \
          return impl::NAMESPACE::OPERATOR_NAME<scalar_t>(                      \
            ARG1,                                                               \
            ARG2,                                                               \
            ARG3                                                                \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return iterator.output();                                                     \
}                                                                               \
                                                                                \
inline std::tuple<                                                              \
  at::Tensor,                                                                   \
  at::Tensor,                                                                   \
  at::Tensor                                                                    \
> OPERATOR_NAME##_backward(                                                     \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3                                                        \
) {                                                                             \
  at::Tensor gradient_##ARG1;                                                   \
  at::Tensor gradient_##ARG2;                                                   \
  at::Tensor gradient_##ARG3;                                                   \
                                                                                \
  auto iterator = at::TensorIteratorConfig()                                    \
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
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_backward_cuda",                                            \
    [&]() {                                                                     \
      at::native::gpu_kernel_multiple_outputs(                                  \
        iterator,                                                               \
        []GPU_LAMBDA(                                                           \
          scalar_t gradient_output,                                             \
          scalar_t ARG1,                                                        \
          scalar_t ARG2,                                                        \
          scalar_t ARG3                                                         \
        ) -> std::tuple<                                                     \
          scalar_t,                                                             \
          scalar_t,                                                             \
          scalar_t                                                              \
        > {                                                                     \
          auto result =                                                         \
            impl::NAMESPACE::OPERATOR_NAME##_backward<scalar_t>(                \
              gradient_output,                                                  \
              ARG1,                                                             \
              ARG2,                                                             \
              ARG3                                                              \
            );                                                                  \
          return std::make_tuple(                                            \
            std::get<0>(result),                                                \
            std::get<1>(result),                                                \
            std::get<2>(result)                                                 \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return std::make_tuple(                                                       \
    iterator.output(0),                                                         \
    iterator.output(1),                                                         \
    iterator.output(2)                                                          \
  );                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<                                                              \
  at::Tensor,                                                                   \
  at::Tensor,                                                                   \
  at::Tensor,                                                                   \
  at::Tensor                                                                    \
> OPERATOR_NAME##_backward_backward(                                            \
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
    return std::make_tuple(                                                     \
      at::Tensor(),                                                             \
      at::Tensor(),                                                             \
      at::Tensor(),                                                             \
      at::Tensor()                                                              \
    );                                                                          \
  }                                                                             \
                                                                                \
  at::Tensor gradient_gradient_output;                                          \
  at::Tensor gradient_##ARG1;                                                   \
  at::Tensor gradient_##ARG2;                                                   \
  at::Tensor gradient_##ARG3;                                                   \
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
  auto iterator = at::TensorIteratorConfig()                                    \
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
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_backward_backward_cuda",                                   \
    [&]() {                                                                     \
      at::native::gpu_kernel_multiple_outputs(                                  \
        iterator,                                                               \
        [                                                                       \
          has_gradient_gradient_##ARG1,                                         \
          has_gradient_gradient_##ARG2,                                         \
          has_gradient_gradient_##ARG3                                          \
        ]GPU_LAMBDA(                                                            \
          scalar_t gradient_gradient_##ARG1,                                    \
          scalar_t gradient_gradient_##ARG2,                                    \
          scalar_t gradient_gradient_##ARG3,                                    \
          scalar_t gradient_output,                                             \
          scalar_t ARG1,                                                        \
          scalar_t ARG2,                                                        \
          scalar_t ARG3                                                         \
        ) -> std::tuple<                                                     \
          scalar_t,                                                             \
          scalar_t,                                                             \
          scalar_t,                                                             \
          scalar_t                                                              \
        > {                                                                     \
          auto result =                                                         \
            impl::NAMESPACE::OPERATOR_NAME##_backward_backward<scalar_t>(       \
              gradient_gradient_##ARG1,                                         \
              gradient_gradient_##ARG2,                                         \
              gradient_gradient_##ARG3,                                         \
              gradient_output,                                                  \
              ARG1,                                                             \
              ARG2,                                                             \
              ARG3,                                                             \
              has_gradient_gradient_##ARG1,                                     \
              has_gradient_gradient_##ARG2,                                     \
              has_gradient_gradient_##ARG3                                      \
            );                                                                  \
          return std::make_tuple(                                            \
            std::get<0>(result),                                                \
            std::get<1>(result),                                                \
            std::get<2>(result),                                                \
            std::get<3>(result)                                                 \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return std::make_tuple(                                                       \
    iterator.output(0),                                                         \
    iterator.output(1),                                                         \
    iterator.output(2),                                                         \
    iterator.output(3)                                                          \
  );                                                                            \
}                                                                               \
                                                                                \
}  /* namespace torchscience::cuda::NAMESPACE */                                \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {                                \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME                               \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME##_backward                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME##_backward_backward           \
  );                                                                            \
}

#define CUDA_QUATERNARY_OPERATOR(                                               \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3,                                                                         \
  ARG4                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::cuda::NAMESPACE {                                       \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3,                                                       \
  const at::Tensor& ARG4                                                        \
) {                                                                             \
  at::Tensor output;                                                            \
                                                                                \
  auto iterator = at::TensorIteratorConfig()                                    \
    .add_output(output)                                                         \
    .add_const_input(ARG1)                                                      \
    .add_const_input(ARG2)                                                      \
    .add_const_input(ARG3)                                                      \
    .add_const_input(ARG4)                                                      \
    .promote_inputs_to_common_dtype(true)                                       \
    .cast_common_dtype_to_outputs(true)                                         \
    .enforce_safe_casting_to_output(false)                                      \
    .build();                                                                   \
                                                                                \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_cuda",                                                     \
    [&]() {                                                                     \
      at::native::gpu_kernel(                                                   \
        iterator,                                                               \
        []GPU_LAMBDA(                                                           \
          scalar_t ARG1,                                                        \
          scalar_t ARG2,                                                        \
          scalar_t ARG3,                                                        \
          scalar_t ARG4                                                         \
        ) -> scalar_t {                                                         \
          return impl::NAMESPACE::OPERATOR_NAME<scalar_t>(                      \
            ARG1,                                                               \
            ARG2,                                                               \
            ARG3,                                                               \
            ARG4                                                                \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return iterator.output();                                                     \
}                                                                               \
                                                                                \
inline std::tuple<                                                              \
  at::Tensor,                                                                   \
  at::Tensor,                                                                   \
  at::Tensor,                                                                   \
  at::Tensor                                                                    \
> OPERATOR_NAME##_backward(                                                     \
  const at::Tensor& gradient_output,                                            \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3,                                                       \
  const at::Tensor& ARG4                                                        \
) {                                                                             \
  at::Tensor gradient_##ARG1;                                                   \
  at::Tensor gradient_##ARG2;                                                   \
  at::Tensor gradient_##ARG3;                                                   \
  at::Tensor gradient_##ARG4;                                                   \
                                                                                \
  auto iterator = at::TensorIteratorConfig()                                    \
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
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_backward_cuda",                                            \
    [&]() {                                                                     \
      at::native::gpu_kernel_multiple_outputs(                                  \
        iterator,                                                               \
        []GPU_LAMBDA(                                                           \
          scalar_t gradient_output,                                             \
          scalar_t ARG1,                                                        \
          scalar_t ARG2,                                                        \
          scalar_t ARG3,                                                        \
          scalar_t ARG4                                                         \
        ) -> std::tuple<                                                     \
          scalar_t,                                                             \
          scalar_t,                                                             \
          scalar_t,                                                             \
          scalar_t                                                              \
        > {                                                                     \
          auto result =                                                         \
            impl::NAMESPACE::OPERATOR_NAME##_backward<scalar_t>(                \
              gradient_output,                                                  \
              ARG1,                                                             \
              ARG2,                                                             \
              ARG3,                                                             \
              ARG4                                                              \
            );                                                                  \
          return std::make_tuple(                                            \
            std::get<0>(result),                                                \
            std::get<1>(result),                                                \
            std::get<2>(result),                                                \
            std::get<3>(result)                                                 \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return std::make_tuple(                                                       \
    iterator.output(0),                                                         \
    iterator.output(1),                                                         \
    iterator.output(2),                                                         \
    iterator.output(3)                                                          \
  );                                                                            \
}                                                                               \
                                                                                \
inline std::tuple<                                                              \
  at::Tensor,                                                                   \
  at::Tensor,                                                                   \
  at::Tensor,                                                                   \
  at::Tensor,                                                                   \
  at::Tensor                                                                    \
> OPERATOR_NAME##_backward_backward(                                            \
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
    return std::make_tuple(                                                     \
      at::Tensor(),                                                             \
      at::Tensor(),                                                             \
      at::Tensor(),                                                             \
      at::Tensor(),                                                             \
      at::Tensor()                                                              \
    );                                                                          \
  }                                                                             \
                                                                                \
  at::Tensor gradient_gradient_output;                                          \
  at::Tensor gradient_##ARG1;                                                   \
  at::Tensor gradient_##ARG2;                                                   \
  at::Tensor gradient_##ARG3;                                                   \
  at::Tensor gradient_##ARG4;                                                   \
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
  auto iterator = at::TensorIteratorConfig()                                    \
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
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                  \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    iterator.common_dtype(),                                                    \
    #OPERATOR_NAME "_backward_backward_cuda",                                   \
    [&]() {                                                                     \
      at::native::gpu_kernel_multiple_outputs(                                  \
        iterator,                                                               \
        [                                                                       \
          has_gradient_gradient_##ARG1,                                         \
          has_gradient_gradient_##ARG2,                                         \
          has_gradient_gradient_##ARG3,                                         \
          has_gradient_gradient_##ARG4                                          \
        ]GPU_LAMBDA(                                                            \
          scalar_t gradient_gradient_##ARG1,                                    \
          scalar_t gradient_gradient_##ARG2,                                    \
          scalar_t gradient_gradient_##ARG3,                                    \
          scalar_t gradient_gradient_##ARG4,                                    \
          scalar_t gradient_output,                                             \
          scalar_t ARG1,                                                        \
          scalar_t ARG2,                                                        \
          scalar_t ARG3,                                                        \
          scalar_t ARG4                                                         \
        ) -> std::tuple<                                                     \
          scalar_t,                                                             \
          scalar_t,                                                             \
          scalar_t,                                                             \
          scalar_t,                                                             \
          scalar_t                                                              \
        > {                                                                     \
          auto result =                                                         \
            impl::NAMESPACE::OPERATOR_NAME##_backward_backward<scalar_t>(       \
              gradient_gradient_##ARG1,                                         \
              gradient_gradient_##ARG2,                                         \
              gradient_gradient_##ARG3,                                         \
              gradient_gradient_##ARG4,                                         \
              gradient_output,                                                  \
              ARG1,                                                             \
              ARG2,                                                             \
              ARG3,                                                             \
              ARG4,                                                             \
              has_gradient_gradient_##ARG1,                                     \
              has_gradient_gradient_##ARG2,                                     \
              has_gradient_gradient_##ARG3,                                     \
              has_gradient_gradient_##ARG4                                      \
            );                                                                  \
          return std::make_tuple(                                            \
            std::get<0>(result),                                                \
            std::get<1>(result),                                                \
            std::get<2>(result),                                                \
            std::get<3>(result),                                                \
            std::get<4>(result)                                                 \
          );                                                                    \
        }                                                                       \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  return std::make_tuple(                                                       \
    iterator.output(0),                                                         \
    iterator.output(1),                                                         \
    iterator.output(2),                                                         \
    iterator.output(3),                                                         \
    iterator.output(4)                                                          \
  );                                                                            \
}                                                                               \
                                                                                \
}  /* namespace torchscience::cuda::NAMESPACE */                                \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {                                \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME                               \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward",                                                 \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME##_backward                    \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #OPERATOR_NAME "_backward_backward",                                        \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME##_backward_backward           \
  );                                                                            \
}
