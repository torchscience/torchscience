#pragma once

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "../core/creation_common.h"

#ifndef TORCHSCIENCE_UNPACK_IMPL
#define TORCHSCIENCE_UNPACK_IMPL(...) __VA_ARGS__
#define TORCHSCIENCE_UNPACK(X) TORCHSCIENCE_UNPACK_IMPL X
#define TORCHSCIENCE_COMMA_IF_IMPL(...) __VA_OPT__(,)
#define TORCHSCIENCE_COMMA_IF(X) TORCHSCIENCE_COMMA_IF_IMPL(TORCHSCIENCE_UNPACK(X))
#endif

#define CUDA_CREATION_OPERATOR(                                                 \
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
  const c10::optional<at::ScalarType>& dtype,                                   \
  const c10::optional<at::Layout>& layout,                                      \
  const c10::optional<at::Device>& device,                                      \
  bool requires_grad                                                            \
) {                                                                             \
  auto dev = device.value_or(at::kCUDA);                                        \
  TORCH_CHECK(dev.is_cuda(), #OPERATOR_NAME ": device must be CUDA");           \
                                                                                \
  /* CUDA device guard - critical for multi-GPU */                              \
  c10::cuda::CUDAGuard guard(dev);                                              \
                                                                                \
  std::vector<int64_t> shape_vec = OUTPUT_SHAPE;                                \
  ::torchscience::core::check_size_nonnegative(shape_vec, #OPERATOR_NAME);      \
                                                                                \
  auto options = ::torchscience::core::build_options(                           \
    dtype, layout, device, dev                                                  \
  );                                                                            \
                                                                                \
  int64_t numel = ::torchscience::core::compute_numel(shape_vec);               \
                                                                                \
  at::Tensor output = at::empty(shape_vec, options);                            \
                                                                                \
  if (numel > 0) {                                                              \
    const int threads = 256;                                                    \
    const int blocks = (numel + threads - 1) / threads;                         \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                            \
      at::kBFloat16,                                                            \
      at::kHalf,                                                                \
      output.scalar_type(),                                                     \
      #OPERATOR_NAME "_cuda",                                                   \
      [&]() {                                                                   \
        impl::NAMESPACE::OPERATOR_NAME##_kernel<scalar_t><<<blocks, threads>>>( \
          output.data_ptr<scalar_t>(),                                          \
          numel                                                                 \
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
