#pragma once

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>
#include <torch/library.h>

// =============================================================================
// Helper macros for handling parenthesized parameter lists
// =============================================================================

#ifndef TORCHSCIENCE_UNPACK_IMPL
#define TORCHSCIENCE_UNPACK_IMPL(...) __VA_ARGS__
#define TORCHSCIENCE_UNPACK(X) TORCHSCIENCE_UNPACK_IMPL X
#define TORCHSCIENCE_COMMA_IF_IMPL(...) __VA_OPT__(, __VA_ARGS__)
#define TORCHSCIENCE_COMMA_IF(X) TORCHSCIENCE_COMMA_IF_IMPL(TORCHSCIENCE_UNPACK(X))
#endif

// =============================================================================
// CUDA_STOCHASTIC_CREATION_OPERATOR
// =============================================================================
// Flexible macro for creating tensors with random elements on CUDA.
// Uses philox RNG for reproducibility.
//
// Kernel signature expected:
//   namespace impl::NAMESPACE {
//     template <typename scalar_t>
//     __global__ void OPERATOR_NAME_kernel(
//       scalar_t* output, int64_t numel,
//       at::PhiloxCudaState philox_args, PARAMS...
//     )
//   }
// =============================================================================

#define CUDA_STOCHASTIC_CREATION_OPERATOR(                                      \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  OUTPUT_SHAPE,                                                                 \
  PARAMS,                                                                       \
  ARGS                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::cuda::NAMESPACE {                                       \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  TORCHSCIENCE_UNPACK(PARAMS)                                                   \
  TORCHSCIENCE_COMMA_IF(PARAMS)                                                 \
  c10::optional<at::Generator> generator,                                       \
  const c10::optional<at::ScalarType>& dtype,                                   \
  const c10::optional<at::Layout>& layout,                                      \
  const c10::optional<at::Device>& device,                                      \
  bool requires_grad                                                            \
) {                                                                             \
  auto dev = device.value_or(at::kCUDA);                                        \
  TORCH_CHECK(dev.is_cuda(), #OPERATOR_NAME ": device must be CUDA");           \
                                                                                \
  c10::cuda::CUDAGuard guard(dev);                                              \
                                                                                \
  auto options = at::TensorOptions()                                            \
    .dtype(dtype.value_or(                                                      \
      c10::typeMetaToScalarType(at::get_default_dtype())                        \
    ))                                                                          \
    .layout(layout.value_or(at::kStrided))                                      \
    .device(dev)                                                                \
    .requires_grad(false);                                                      \
                                                                                \
  std::vector<int64_t> shape_vec = OUTPUT_SHAPE;                                \
  for (auto s : shape_vec) {                                                    \
    TORCH_CHECK(s >= 0,                                                         \
      #OPERATOR_NAME ": size must be non-negative, got ", s);                   \
  }                                                                             \
                                                                                \
  int64_t numel = 1;                                                            \
  for (auto s : shape_vec) {                                                    \
    numel *= s;                                                                 \
  }                                                                             \
                                                                                \
  at::Tensor output = at::empty(shape_vec, options);                            \
                                                                                \
  if (numel > 0) {                                                              \
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(             \
      generator, at::cuda::detail::getDefaultCUDAGenerator()                    \
    );                                                                          \
                                                                                \
    const int threads = 256;                                                    \
    const int blocks = (numel + threads - 1) / threads;                         \
                                                                                \
    at::PhiloxCudaState philox_args;                                            \
    {                                                                           \
      std::lock_guard<std::mutex> lock(gen->mutex_);                            \
      philox_args = gen->philox_cuda_state(numel);                              \
    }                                                                           \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                            \
      at::kBFloat16,                                                            \
      at::kHalf,                                                                \
      output.scalar_type(),                                                     \
      #OPERATOR_NAME "_cuda",                                                   \
      [&]() {                                                                   \
        impl::NAMESPACE::OPERATOR_NAME##_kernel<scalar_t><<<blocks, threads>>>( \
          output.data_ptr<scalar_t>(),                                          \
          numel,                                                                \
          philox_args                                                           \
          TORCHSCIENCE_COMMA_IF(ARGS)                                           \
          TORCHSCIENCE_UNPACK(ARGS)                                             \
        );                                                                      \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
      }                                                                         \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (requires_grad) {                                                          \
    output = output.requires_grad_(true);                                       \
  }                                                                             \
                                                                                \
  return output;                                                                \
}                                                                               \
                                                                                \
}  /* namespace torchscience::cuda::NAMESPACE */                                \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {                                \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::cuda::NAMESPACE::OPERATOR_NAME                               \
  );                                                                            \
}
