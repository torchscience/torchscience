#pragma once

#include <tuple>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <thrust/tuple.h>
#include <torch/library.h>

// Promote c10::Half/BFloat16 to float for CUDA kernel computation.
// NVCC has ambiguous comparison operators between c10::Half and
// cuda_fp16.hpp's __half operators — promoting to float avoids this.
namespace torchscience::cuda {
template <typename T> struct promote { using type = T; };
template <> struct promote<c10::Half> { using type = float; };
template <> struct promote<c10::BFloat16> { using type = float; };
template <typename T> using promote_t = typename promote<T>::type;
} // namespace torchscience::cuda

#define TORCHSCIENCE_CUDA_POINTWISE_UNARY_DISPATCH(category, name, arg1)                             \
namespace torchscience::cuda::category {                               \
                                                          \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1##_input                                                \
) {                                                                            \
  c10::cuda::CUDAGuard device_guard(arg1##_input.device());                     \
                                                                               \
  at::Tensor output;                                                           \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(output)                                                        \
    .add_const_input(arg1##_input)                                              \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name,                                                                     \
    [&] {                                                                      \
      at::native::gpu_kernel(                                                  \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t arg1                                                         \
        ) -> scalar_t {                                                        \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          return static_cast<scalar_t>(kernel::category::name(       \
            static_cast<compute_t>(arg1)                                       \
          ));                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return iterator.output();                                                    \
}                                                                              \
                                                                               \
inline at::Tensor name##_backward(                                             \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input                                                \
) {                                                                            \
  c10::cuda::CUDAGuard device_guard(gradient_input.device());                   \
                                                                               \
  at::Tensor gradient_output;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_output)                                               \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                              \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward",                                                         \
    [&] {                                                                      \
      at::native::gpu_kernel(                                                  \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t gradient,                                                   \
          scalar_t arg1                                                         \
        ) -> scalar_t {                                                        \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          return static_cast<scalar_t>(kernel::category::name##_backward( \
            static_cast<compute_t>(gradient),                                  \
            static_cast<compute_t>(arg1)                                       \
          ));                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return iterator.output();                                                    \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(            \
  const at::Tensor &gradient_gradient_input,                                   \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input                                                \
) {                                                                            \
  if (!gradient_gradient_input.defined()) {                                    \
    return {                                                                   \
      at::Tensor(),                                                            \
      at::Tensor()                                                             \
    };                                                                         \
  }                                                                            \
                                                                               \
  c10::cuda::CUDAGuard device_guard(gradient_gradient_input.device());          \
                                                                               \
  at::Tensor gradient_gradient_output;                                         \
  at::Tensor gradient_output;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_gradient_output)                                      \
    .add_output(gradient_output)                                               \
    .add_const_input(gradient_gradient_input)                                  \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                              \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward_backward",                                                \
    [&] {                                                                      \
      at::native::gpu_kernel_multiple_outputs(                                 \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t gradient_gradient,                                          \
          scalar_t gradient,                                                   \
          scalar_t arg1                                                         \
        ) -> thrust::tuple<                                                    \
          scalar_t,                                                            \
          scalar_t                                                             \
        > {                                                                    \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          auto result = kernel::category::name##_backward_backward(   \
            static_cast<compute_t>(gradient_gradient),                         \
            static_cast<compute_t>(gradient),                                  \
            static_cast<compute_t>(arg1)                                       \
          );                                                                   \
          return thrust::make_tuple(                                           \
            static_cast<scalar_t>(std::get<0>(result)),                        \
            static_cast<scalar_t>(std::get<1>(result))                         \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::cuda::category */                         \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {                                \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::cuda::category::name                                 \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::cuda::category::name##_backward                      \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::cuda::category::name##_backward_backward             \
  );                                                                           \
}

#define TORCHSCIENCE_CUDA_POINTWISE_BINARY_DISPATCH(category, name, arg1, arg2)                     \
namespace torchscience::cuda::category {                               \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input                                               \
) {                                                                            \
  c10::cuda::CUDAGuard device_guard(arg1##_input.device());                     \
                                                                               \
  at::Tensor output;                                                           \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(output)                                                        \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name,                                                                     \
    [&] {                                                                      \
      at::native::gpu_kernel(                                                  \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t arg1,                                                       \
          scalar_t arg2                                                        \
        ) -> scalar_t {                                                        \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          return static_cast<scalar_t>(kernel::category::name(       \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2)                                       \
          ));                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return iterator.output();                                                    \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor> name##_backward(                     \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input                                               \
) {                                                                            \
  c10::cuda::CUDAGuard device_guard(gradient_input.device());                   \
                                                                               \
  at::Tensor arg1##_gradient_output;                                           \
  at::Tensor arg2##_gradient_output;                                           \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(arg1##_gradient_output)                                        \
    .add_output(arg2##_gradient_output)                                        \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward",                                                         \
    [&] {                                                                      \
      at::native::gpu_kernel_multiple_outputs(                                 \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t gradient,                                                   \
          scalar_t arg1,                                                       \
          scalar_t arg2                                                        \
        ) -> thrust::tuple<scalar_t, scalar_t> {                               \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          auto result = kernel::category::name##_backward(            \
            static_cast<compute_t>(gradient),                                  \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2)                                       \
          );                                                                   \
          return thrust::make_tuple(                                           \
            static_cast<scalar_t>(std::get<0>(result)),                        \
            static_cast<scalar_t>(std::get<1>(result))                         \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> name##_backward_backward(\
  const at::Tensor &arg1##_gradient_gradient_input,                            \
  const at::Tensor &arg2##_gradient_gradient_input,                            \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input                                               \
) {                                                                            \
  if (!arg1##_gradient_gradient_input.defined() &&                             \
      !arg2##_gradient_gradient_input.defined()) {                             \
    return {                                                                   \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor()                                                             \
    };                                                                         \
  }                                                                            \
                                                                               \
  c10::cuda::CUDAGuard device_guard(gradient_input.device());                   \
                                                                               \
  at::Tensor gradient_gradient_output;                                         \
  at::Tensor arg1##_gradient_output;                                           \
  at::Tensor arg2##_gradient_output;                                           \
                                                                               \
  auto arg1##_gg = arg1##_gradient_gradient_input.defined()                    \
    ? arg1##_gradient_gradient_input                                           \
    : at::zeros_like(arg1##_input);                                            \
  auto arg2##_gg = arg2##_gradient_gradient_input.defined()                    \
    ? arg2##_gradient_gradient_input                                           \
    : at::zeros_like(arg2##_input);                                            \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_gradient_output)                                      \
    .add_output(arg1##_gradient_output)                                        \
    .add_output(arg2##_gradient_output)                                        \
    .add_const_input(arg1##_gg)                                                \
    .add_const_input(arg2##_gg)                                                \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward_backward",                                                \
    [&] {                                                                      \
      at::native::gpu_kernel_multiple_outputs(                                 \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t arg1##_gradient_gradient,                                   \
          scalar_t arg2##_gradient_gradient,                                   \
          scalar_t gradient,                                                   \
          scalar_t arg1,                                                       \
          scalar_t arg2                                                        \
        ) -> thrust::tuple<scalar_t, scalar_t, scalar_t> {                     \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          auto result = kernel::category::name##_backward_backward(   \
            static_cast<compute_t>(arg1##_gradient_gradient),                  \
            static_cast<compute_t>(arg2##_gradient_gradient),                  \
            static_cast<compute_t>(gradient),                                  \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2)                                       \
          );                                                                   \
          return thrust::make_tuple(                                           \
            static_cast<scalar_t>(std::get<0>(result)),                        \
            static_cast<scalar_t>(std::get<1>(result)),                        \
            static_cast<scalar_t>(std::get<2>(result))                         \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::cuda::category */                         \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {                                \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::cuda::category::name                                 \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::cuda::category::name##_backward                      \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::cuda::category::name##_backward_backward             \
  );                                                                           \
}

#define TORCHSCIENCE_CUDA_POINTWISE_TERNARY_DISPATCH(category, name, arg1, arg2, arg3)              \
namespace torchscience::cuda::category {                               \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input                                               \
) {                                                                            \
  c10::cuda::CUDAGuard device_guard(arg1##_input.device());                     \
                                                                               \
  at::Tensor output;                                                           \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(output)                                                        \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name,                                                                     \
    [&] {                                                                      \
      at::native::gpu_kernel(                                                  \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t arg1,                                                       \
          scalar_t arg2,                                                       \
          scalar_t arg3                                                        \
        ) -> scalar_t {                                                        \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          return static_cast<scalar_t>(kernel::category::name(       \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2),                                      \
            static_cast<compute_t>(arg3)                                       \
          ));                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return iterator.output();                                                    \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> name##_backward(         \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input                                               \
) {                                                                            \
  c10::cuda::CUDAGuard device_guard(gradient_input.device());                   \
                                                                               \
  at::Tensor arg1##_gradient_output;                                           \
  at::Tensor arg2##_gradient_output;                                           \
  at::Tensor arg3##_gradient_output;                                           \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(arg1##_gradient_output)                                        \
    .add_output(arg2##_gradient_output)                                        \
    .add_output(arg3##_gradient_output)                                        \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward",                                                         \
    [&] {                                                                      \
      at::native::gpu_kernel_multiple_outputs(                                 \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t gradient,                                                   \
          scalar_t arg1,                                                       \
          scalar_t arg2,                                                       \
          scalar_t arg3                                                        \
        ) -> thrust::tuple<scalar_t, scalar_t, scalar_t> {                     \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          auto result = kernel::category::name##_backward(            \
            static_cast<compute_t>(gradient),                                  \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2),                                      \
            static_cast<compute_t>(arg3)                                       \
          );                                                                   \
          return thrust::make_tuple(                                           \
            static_cast<scalar_t>(std::get<0>(result)),                        \
            static_cast<scalar_t>(std::get<1>(result)),                        \
            static_cast<scalar_t>(std::get<2>(result))                         \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
inline std::tuple<                                                             \
  at::Tensor, at::Tensor, at::Tensor, at::Tensor                               \
> name##_backward_backward(                                                    \
  const at::Tensor &arg1##_gradient_gradient_input,                            \
  const at::Tensor &arg2##_gradient_gradient_input,                            \
  const at::Tensor &arg3##_gradient_gradient_input,                            \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input                                               \
) {                                                                            \
  if (!arg1##_gradient_gradient_input.defined() &&                             \
      !arg2##_gradient_gradient_input.defined() &&                             \
      !arg3##_gradient_gradient_input.defined()) {                             \
    return {                                                                   \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor()                                                             \
    };                                                                         \
  }                                                                            \
                                                                               \
  c10::cuda::CUDAGuard device_guard(gradient_input.device());                   \
                                                                               \
  at::Tensor gradient_gradient_output;                                         \
  at::Tensor arg1##_gradient_output;                                           \
  at::Tensor arg2##_gradient_output;                                           \
  at::Tensor arg3##_gradient_output;                                           \
                                                                               \
  auto arg1##_gg = arg1##_gradient_gradient_input.defined()                    \
    ? arg1##_gradient_gradient_input                                           \
    : at::zeros_like(arg1##_input);                                            \
  auto arg2##_gg = arg2##_gradient_gradient_input.defined()                    \
    ? arg2##_gradient_gradient_input                                           \
    : at::zeros_like(arg2##_input);                                            \
  auto arg3##_gg = arg3##_gradient_gradient_input.defined()                    \
    ? arg3##_gradient_gradient_input                                           \
    : at::zeros_like(arg3##_input);                                            \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_gradient_output)                                      \
    .add_output(arg1##_gradient_output)                                        \
    .add_output(arg2##_gradient_output)                                        \
    .add_output(arg3##_gradient_output)                                        \
    .add_const_input(arg1##_gg)                                                \
    .add_const_input(arg2##_gg)                                                \
    .add_const_input(arg3##_gg)                                                \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward_backward",                                                \
    [&] {                                                                      \
      at::native::gpu_kernel_multiple_outputs(                                 \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t arg1##_gradient_gradient,                                   \
          scalar_t arg2##_gradient_gradient,                                   \
          scalar_t arg3##_gradient_gradient,                                   \
          scalar_t gradient,                                                   \
          scalar_t arg1,                                                       \
          scalar_t arg2,                                                       \
          scalar_t arg3                                                        \
        ) -> thrust::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {           \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          auto result = kernel::category::name##_backward_backward(   \
            static_cast<compute_t>(arg1##_gradient_gradient),                  \
            static_cast<compute_t>(arg2##_gradient_gradient),                  \
            static_cast<compute_t>(arg3##_gradient_gradient),                  \
            static_cast<compute_t>(gradient),                                  \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2),                                      \
            static_cast<compute_t>(arg3)                                       \
          );                                                                   \
          return thrust::make_tuple(                                           \
            static_cast<scalar_t>(std::get<0>(result)),                        \
            static_cast<scalar_t>(std::get<1>(result)),                        \
            static_cast<scalar_t>(std::get<2>(result)),                        \
            static_cast<scalar_t>(std::get<3>(result))                         \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2),                                                        \
    iterator.output(3)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::cuda::category */                         \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {                                \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::cuda::category::name                                 \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::cuda::category::name##_backward                      \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::cuda::category::name##_backward_backward             \
  );                                                                           \
}

#define TORCHSCIENCE_CUDA_POINTWISE_QUATERNARY_DISPATCH(category, name, arg1, arg2, arg3, arg4)     \
namespace torchscience::cuda::category {                               \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input                                               \
) {                                                                            \
  c10::cuda::CUDAGuard device_guard(arg1##_input.device());                     \
                                                                               \
  at::Tensor output;                                                           \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(output)                                                        \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name,                                                                     \
    [&] {                                                                      \
      at::native::gpu_kernel(                                                  \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t arg1,                                                       \
          scalar_t arg2,                                                       \
          scalar_t arg3,                                                       \
          scalar_t arg4                                                        \
        ) -> scalar_t {                                                        \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          return static_cast<scalar_t>(kernel::category::name(       \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2),                                      \
            static_cast<compute_t>(arg3),                                      \
            static_cast<compute_t>(arg4)                                       \
          ));                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return iterator.output();                                                    \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>              \
name##_backward(                                                               \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input                                               \
) {                                                                            \
  c10::cuda::CUDAGuard device_guard(gradient_input.device());                   \
                                                                               \
  at::Tensor arg1##_gradient_output;                                           \
  at::Tensor arg2##_gradient_output;                                           \
  at::Tensor arg3##_gradient_output;                                           \
  at::Tensor arg4##_gradient_output;                                           \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(arg1##_gradient_output)                                        \
    .add_output(arg2##_gradient_output)                                        \
    .add_output(arg3##_gradient_output)                                        \
    .add_output(arg4##_gradient_output)                                        \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward",                                                         \
    [&] {                                                                      \
      at::native::gpu_kernel_multiple_outputs(                                 \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t gradient,                                                   \
          scalar_t arg1,                                                       \
          scalar_t arg2,                                                       \
          scalar_t arg3,                                                       \
          scalar_t arg4                                                        \
        ) -> thrust::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {           \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          auto result = kernel::category::name##_backward(            \
            static_cast<compute_t>(gradient),                                  \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2),                                      \
            static_cast<compute_t>(arg3),                                      \
            static_cast<compute_t>(arg4)                                       \
          );                                                                   \
          return thrust::make_tuple(                                           \
            static_cast<scalar_t>(std::get<0>(result)),                        \
            static_cast<scalar_t>(std::get<1>(result)),                        \
            static_cast<scalar_t>(std::get<2>(result)),                        \
            static_cast<scalar_t>(std::get<3>(result))                         \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2),                                                        \
    iterator.output(3)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
inline std::tuple<                                                             \
  at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor                   \
> name##_backward_backward(                                                    \
  const at::Tensor &arg1##_gradient_gradient_input,                            \
  const at::Tensor &arg2##_gradient_gradient_input,                            \
  const at::Tensor &arg3##_gradient_gradient_input,                            \
  const at::Tensor &arg4##_gradient_gradient_input,                            \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input                                               \
) {                                                                            \
  if (!arg1##_gradient_gradient_input.defined() &&                             \
      !arg2##_gradient_gradient_input.defined() &&                             \
      !arg3##_gradient_gradient_input.defined() &&                             \
      !arg4##_gradient_gradient_input.defined()) {                             \
    return {                                                                   \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor()                                                             \
    };                                                                         \
  }                                                                            \
                                                                               \
  c10::cuda::CUDAGuard device_guard(gradient_input.device());                   \
                                                                               \
  at::Tensor gradient_gradient_output;                                         \
  at::Tensor arg1##_gradient_output;                                           \
  at::Tensor arg2##_gradient_output;                                           \
  at::Tensor arg3##_gradient_output;                                           \
  at::Tensor arg4##_gradient_output;                                           \
                                                                               \
  auto arg1##_gg = arg1##_gradient_gradient_input.defined()                    \
    ? arg1##_gradient_gradient_input                                           \
    : at::zeros_like(arg1##_input);                                            \
  auto arg2##_gg = arg2##_gradient_gradient_input.defined()                    \
    ? arg2##_gradient_gradient_input                                           \
    : at::zeros_like(arg2##_input);                                            \
  auto arg3##_gg = arg3##_gradient_gradient_input.defined()                    \
    ? arg3##_gradient_gradient_input                                           \
    : at::zeros_like(arg3##_input);                                            \
  auto arg4##_gg = arg4##_gradient_gradient_input.defined()                    \
    ? arg4##_gradient_gradient_input                                           \
    : at::zeros_like(arg4##_input);                                            \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_gradient_output)                                      \
    .add_output(arg1##_gradient_output)                                        \
    .add_output(arg2##_gradient_output)                                        \
    .add_output(arg3##_gradient_output)                                        \
    .add_output(arg4##_gradient_output)                                        \
    .add_const_input(arg1##_gg)                                                \
    .add_const_input(arg2##_gg)                                                \
    .add_const_input(arg3##_gg)                                                \
    .add_const_input(arg4##_gg)                                                \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward_backward",                                                \
    [&] {                                                                      \
      at::native::gpu_kernel_multiple_outputs(                                 \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t arg1##_gradient_gradient,                                   \
          scalar_t arg2##_gradient_gradient,                                   \
          scalar_t arg3##_gradient_gradient,                                   \
          scalar_t arg4##_gradient_gradient,                                   \
          scalar_t gradient,                                                   \
          scalar_t arg1,                                                       \
          scalar_t arg2,                                                       \
          scalar_t arg3,                                                       \
          scalar_t arg4                                                        \
        ) -> thrust::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t> { \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          auto result = kernel::category::name##_backward_backward(   \
            static_cast<compute_t>(arg1##_gradient_gradient),                  \
            static_cast<compute_t>(arg2##_gradient_gradient),                  \
            static_cast<compute_t>(arg3##_gradient_gradient),                  \
            static_cast<compute_t>(arg4##_gradient_gradient),                  \
            static_cast<compute_t>(gradient),                                  \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2),                                      \
            static_cast<compute_t>(arg3),                                      \
            static_cast<compute_t>(arg4)                                       \
          );                                                                   \
          return thrust::make_tuple(                                           \
            static_cast<scalar_t>(std::get<0>(result)),                        \
            static_cast<scalar_t>(std::get<1>(result)),                        \
            static_cast<scalar_t>(std::get<2>(result)),                        \
            static_cast<scalar_t>(std::get<3>(result)),                        \
            static_cast<scalar_t>(std::get<4>(result))                         \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2),                                                        \
    iterator.output(3),                                                        \
    iterator.output(4)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::cuda::category */                         \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {                                \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::cuda::category::name                                 \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::cuda::category::name##_backward                      \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::cuda::category::name##_backward_backward             \
  );                                                                           \
}

#define TORCHSCIENCE_CUDA_POINTWISE_QUINARY_DISPATCH(category, name, arg1, arg2, arg3, arg4, arg5)     \
namespace torchscience::cuda::category {                               \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input,                                              \
  const at::Tensor &arg5##_input                                               \
) {                                                                            \
  c10::cuda::CUDAGuard device_guard(arg1##_input.device());                     \
                                                                               \
  at::Tensor output;                                                           \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(output)                                                        \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .add_const_input(arg5##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name,                                                                     \
    [&] {                                                                      \
      at::native::gpu_kernel(                                                  \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t arg1,                                                       \
          scalar_t arg2,                                                       \
          scalar_t arg3,                                                       \
          scalar_t arg4,                                                       \
          scalar_t arg5                                                        \
        ) -> scalar_t {                                                        \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          return static_cast<scalar_t>(kernel::category::name(       \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2),                                      \
            static_cast<compute_t>(arg3),                                      \
            static_cast<compute_t>(arg4),                                      \
            static_cast<compute_t>(arg5)                                       \
          ));                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return iterator.output();                                                    \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>  \
name##_backward(                                                               \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input,                                              \
  const at::Tensor &arg5##_input                                               \
) {                                                                            \
  c10::cuda::CUDAGuard device_guard(gradient_input.device());                   \
                                                                               \
  at::Tensor arg1##_gradient_output;                                           \
  at::Tensor arg2##_gradient_output;                                           \
  at::Tensor arg3##_gradient_output;                                           \
  at::Tensor arg4##_gradient_output;                                           \
  at::Tensor arg5##_gradient_output;                                           \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(arg1##_gradient_output)                                        \
    .add_output(arg2##_gradient_output)                                        \
    .add_output(arg3##_gradient_output)                                        \
    .add_output(arg4##_gradient_output)                                        \
    .add_output(arg5##_gradient_output)                                        \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .add_const_input(arg5##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward",                                                         \
    [&] {                                                                      \
      at::native::gpu_kernel_multiple_outputs(                                 \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t gradient,                                                   \
          scalar_t arg1,                                                       \
          scalar_t arg2,                                                       \
          scalar_t arg3,                                                       \
          scalar_t arg4,                                                       \
          scalar_t arg5                                                        \
        ) -> thrust::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t> { \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          auto result = kernel::category::name##_backward(            \
            static_cast<compute_t>(gradient),                                  \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2),                                      \
            static_cast<compute_t>(arg3),                                      \
            static_cast<compute_t>(arg4),                                      \
            static_cast<compute_t>(arg5)                                       \
          );                                                                   \
          return thrust::make_tuple(                                           \
            static_cast<scalar_t>(std::get<0>(result)),                        \
            static_cast<scalar_t>(std::get<1>(result)),                        \
            static_cast<scalar_t>(std::get<2>(result)),                        \
            static_cast<scalar_t>(std::get<3>(result)),                        \
            static_cast<scalar_t>(std::get<4>(result))                         \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2),                                                        \
    iterator.output(3),                                                        \
    iterator.output(4)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
inline std::tuple<                                                             \
  at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor       \
> name##_backward_backward(                                                    \
  const at::Tensor &arg1##_gradient_gradient_input,                            \
  const at::Tensor &arg2##_gradient_gradient_input,                            \
  const at::Tensor &arg3##_gradient_gradient_input,                            \
  const at::Tensor &arg4##_gradient_gradient_input,                            \
  const at::Tensor &arg5##_gradient_gradient_input,                            \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input,                                              \
  const at::Tensor &arg5##_input                                               \
) {                                                                            \
  if (!arg1##_gradient_gradient_input.defined() &&                             \
      !arg2##_gradient_gradient_input.defined() &&                             \
      !arg3##_gradient_gradient_input.defined() &&                             \
      !arg4##_gradient_gradient_input.defined() &&                             \
      !arg5##_gradient_gradient_input.defined()) {                             \
    return {                                                                   \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor()                                                             \
    };                                                                         \
  }                                                                            \
                                                                               \
  c10::cuda::CUDAGuard device_guard(gradient_input.device());                   \
                                                                               \
  at::Tensor gradient_gradient_output;                                         \
  at::Tensor arg1##_gradient_output;                                           \
  at::Tensor arg2##_gradient_output;                                           \
  at::Tensor arg3##_gradient_output;                                           \
  at::Tensor arg4##_gradient_output;                                           \
  at::Tensor arg5##_gradient_output;                                           \
                                                                               \
  auto arg1##_gg = arg1##_gradient_gradient_input.defined()                    \
    ? arg1##_gradient_gradient_input                                           \
    : at::zeros_like(arg1##_input);                                            \
  auto arg2##_gg = arg2##_gradient_gradient_input.defined()                    \
    ? arg2##_gradient_gradient_input                                           \
    : at::zeros_like(arg2##_input);                                            \
  auto arg3##_gg = arg3##_gradient_gradient_input.defined()                    \
    ? arg3##_gradient_gradient_input                                           \
    : at::zeros_like(arg3##_input);                                            \
  auto arg4##_gg = arg4##_gradient_gradient_input.defined()                    \
    ? arg4##_gradient_gradient_input                                           \
    : at::zeros_like(arg4##_input);                                            \
  auto arg5##_gg = arg5##_gradient_gradient_input.defined()                    \
    ? arg5##_gradient_gradient_input                                           \
    : at::zeros_like(arg5##_input);                                            \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_gradient_output)                                      \
    .add_output(arg1##_gradient_output)                                        \
    .add_output(arg2##_gradient_output)                                        \
    .add_output(arg3##_gradient_output)                                        \
    .add_output(arg4##_gradient_output)                                        \
    .add_output(arg5##_gradient_output)                                        \
    .add_const_input(arg1##_gg)                                                \
    .add_const_input(arg2##_gg)                                                \
    .add_const_input(arg3##_gg)                                                \
    .add_const_input(arg4##_gg)                                                \
    .add_const_input(arg5##_gg)                                                \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .add_const_input(arg5##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                                 \
    at::kBFloat16,                                                             \
    at::kHalf,                                                                 \
    iterator.common_dtype(),                                                   \
    #name "_backward_backward",                                                \
    [&] {                                                                      \
      at::native::gpu_kernel_multiple_outputs(                                 \
        iterator,                                                              \
        [] GPU_LAMBDA (                                                        \
          scalar_t arg1##_gradient_gradient,                                   \
          scalar_t arg2##_gradient_gradient,                                   \
          scalar_t arg3##_gradient_gradient,                                   \
          scalar_t arg4##_gradient_gradient,                                   \
          scalar_t arg5##_gradient_gradient,                                   \
          scalar_t gradient,                                                   \
          scalar_t arg1,                                                       \
          scalar_t arg2,                                                       \
          scalar_t arg3,                                                       \
          scalar_t arg4,                                                       \
          scalar_t arg5                                                        \
        ) -> thrust::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t, scalar_t> { \
          using compute_t = torchscience::cuda::promote_t<scalar_t>;           \
          auto result = kernel::category::name##_backward_backward(   \
            static_cast<compute_t>(arg1##_gradient_gradient),                  \
            static_cast<compute_t>(arg2##_gradient_gradient),                  \
            static_cast<compute_t>(arg3##_gradient_gradient),                  \
            static_cast<compute_t>(arg4##_gradient_gradient),                  \
            static_cast<compute_t>(arg5##_gradient_gradient),                  \
            static_cast<compute_t>(gradient),                                  \
            static_cast<compute_t>(arg1),                                      \
            static_cast<compute_t>(arg2),                                      \
            static_cast<compute_t>(arg3),                                      \
            static_cast<compute_t>(arg4),                                      \
            static_cast<compute_t>(arg5)                                       \
          );                                                                   \
          return thrust::make_tuple(                                           \
            static_cast<scalar_t>(std::get<0>(result)),                        \
            static_cast<scalar_t>(std::get<1>(result)),                        \
            static_cast<scalar_t>(std::get<2>(result)),                        \
            static_cast<scalar_t>(std::get<3>(result)),                        \
            static_cast<scalar_t>(std::get<4>(result)),                        \
            static_cast<scalar_t>(std::get<5>(result))                         \
          );                                                                   \
        }                                                                      \
      );                                                                       \
    }                                                                          \
  );                                                                           \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2),                                                        \
    iterator.output(3),                                                        \
    iterator.output(4),                                                        \
    iterator.output(5)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::cuda::category */                         \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {                                \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::cuda::category::name                                 \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::cuda::category::name##_backward                      \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::cuda::category::name##_backward_backward             \
  );                                                                           \
}

#define TORCHSCIENCE_CUDA_POINTWISE_UNARY(category, complex, name, arg1) \
    TORCHSCIENCE_CUDA_POINTWISE_UNARY_DISPATCH(category, name, arg1)

#define TORCHSCIENCE_CUDA_POINTWISE_BINARY(category, complex, name, arg1, arg2) \
    TORCHSCIENCE_CUDA_POINTWISE_BINARY_DISPATCH(category, name, arg1, arg2)

#define TORCHSCIENCE_CUDA_POINTWISE_TERNARY(category, complex, name, arg1, arg2, arg3) \
    TORCHSCIENCE_CUDA_POINTWISE_TERNARY_DISPATCH(category, name, arg1, arg2, arg3)

#define TORCHSCIENCE_CUDA_POINTWISE_QUATERNARY(category, complex, name, arg1, arg2, arg3, arg4) \
    TORCHSCIENCE_CUDA_POINTWISE_QUATERNARY_DISPATCH(category, name, arg1, arg2, arg3, arg4)

#define TORCHSCIENCE_CUDA_POINTWISE_QUINARY(category, complex, name, arg1, arg2, arg3, arg4, arg5) \
    TORCHSCIENCE_CUDA_POINTWISE_QUINARY_DISPATCH(category, name, arg1, arg2, arg3, arg4, arg5)
