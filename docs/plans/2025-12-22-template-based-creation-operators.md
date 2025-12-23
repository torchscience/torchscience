# Template-Based Creation Operators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace creation macro system (~1,100 lines across 12 files) with type-safe C++20 templates + minimal macros for string concatenation only.

**Architecture:** Templates handle all operator logic (TensorIterator, dispatch, kernels). Minimal macros handle only `#name` stringification. Uses existing `core/creation_common.h` utilities for validation and options building.

**Tech Stack:** C++20 templates, PyTorch dispatcher, ATen TensorOptions, CUDA kernels

---

## Current State

### Creation Macro Files (12 total, ~1,100 lines)
- `cpu/creation_macros.h` - Dense CPU creation (~75 lines)
- `cuda/creation_macros.cuh` - Dense CUDA creation (~120 lines)
- `meta/creation_macros.h` - Shape inference (~75 lines)
- `autocast/creation_macros.h` - Mixed precision (~95 lines)
- `cpu/stochastic_creation_macros.h` - CPU RNG creation (~120 lines)
- `cuda/stochastic_creation_macros.cuh` - CUDA RNG creation (~130 lines)
- `sparse/coo/cpu/creation_macros.h` - Sparse COO CPU (~115 lines)
- `sparse/coo/cuda/creation_macros.cuh` - Sparse COO CUDA (~115 lines)
- `sparse/csr/cpu/creation_macros.h` - Sparse CSR CPU (~110 lines)
- `sparse/csr/cuda/creation_macros.cuh` - Sparse CSR CUDA (~115 lines)
- `quantized/cpu/creation_macros.h` - Quantized CPU (~110 lines)
- `quantized/cuda/creation_macros.cuh` - Quantized CUDA (~120 lines)

### Existing Infrastructure
- `core/creation_common.h` - Shared utilities (check_size_nonnegative, compute_numel, build_options)

### Current Macro Usage Pattern
```cpp
// cpu/signal_processing/window_function/rectangular_window.h
#include "../../creation_macros.h"

CPU_CREATION_OPERATOR(
  window_function,
  rectangular_window,
  {n},
  (int64_t n),
  (n)
)
```

The macro calls `impl::window_function::rectangular_window_kernel<scalar_t>()`.

## Target State

```cpp
// Templates handle all logic
template<typename KernelTraits>
struct CPUCreationOperator {
    template<typename... Args>
    static at::Tensor forward(Args... args, ...options...);
};

// Minimal macro only for string concatenation
#define REGISTER_CPU_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::cpu::CPUCreationOperator<Traits>::forward<__VA_ARGS__>)

// Usage - Traits define shape, kernel, and dispatch behavior
struct RectangularWindowTraits {
    static std::vector<int64_t> output_shape(int64_t n) { return {n}; }
    template<typename scalar_t>
    static void kernel(scalar_t* out, int64_t numel, int64_t n) { ... }
};

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    REGISTER_CPU_CREATION(module, rectangular_window, RectangularWindowTraits, int64_t);
}
```

---

## Implementation Tasks

### Task 1: Create CPU creation operator template

**Files:**
- Create: `src/torchscience/csrc/cpu/creation_operators.h`

**Step 1: Write the template header**

```cpp
#pragma once

#include <vector>
#include <ATen/Dispatch.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>
#include "../core/creation_common.h"

namespace torchscience::cpu {

// CreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t> static void kernel(scalar_t* out, int64_t numel, params...);

template<typename CreationTraits>
struct CPUCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "creation_op");

        auto options = ::torchscience::core::build_options(dtype, layout, device, at::kCPU);
        int64_t numel = ::torchscience::core::compute_numel(shape_vec);

        at::Tensor output = at::empty(shape_vec, options);

        if (numel > 0) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16,
                at::kHalf,
                output.scalar_type(),
                "cpu_creation",
                [&]() {
                    CreationTraits::template kernel<scalar_t>(
                        output.data_ptr<scalar_t>(),
                        numel,
                        args...
                    );
                }
            );
        }

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

// Minimal macro for string concatenation only
#define REGISTER_CPU_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::cpu::CPUCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::cpu
```

**Step 2: Verify it compiles**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience && python -c "import torch; import torchscience"`
Expected: No compilation errors

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cpu/creation_operators.h
git commit -m "$(cat <<'EOF'
feat: add CPUCreationOperator template

Template-based creation operator infrastructure for CPU.
Uses core/creation_common.h for validation and options.
Templates handle all logic; macros only do string concatenation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Create CUDA creation operator template

**Files:**
- Create: `src/torchscience/csrc/cuda/creation_operators.cuh`

**Step 1: Write the template**

```cpp
#pragma once

#include <vector>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "../core/creation_common.h"

namespace torchscience::cuda {

// CreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t> static void launch_kernel(scalar_t* out, int64_t numel, params...);

template<typename CreationTraits>
struct CUDACreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        auto dev = device.value_or(at::kCUDA);
        TORCH_CHECK(dev.is_cuda(), "device must be CUDA");

        c10::cuda::CUDAGuard guard(dev);

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "cuda_creation_op");

        auto options = ::torchscience::core::build_options(dtype, layout, device, dev);
        int64_t numel = ::torchscience::core::compute_numel(shape_vec);

        at::Tensor output = at::empty(shape_vec, options);

        if (numel > 0) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16,
                at::kHalf,
                output.scalar_type(),
                "cuda_creation",
                [&]() {
                    CreationTraits::template launch_kernel<scalar_t>(
                        output.data_ptr<scalar_t>(),
                        numel,
                        args...
                    );
                }
            );
        }

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_CUDA_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::cuda::CUDACreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::cuda
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cuda/creation_operators.cuh
git commit -m "$(cat <<'EOF'
feat: add CUDACreationOperator template

Template-based creation operator infrastructure for CUDA.
Uses core/creation_common.h for validation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Create Meta creation operator template

**Files:**
- Create: `src/torchscience/csrc/meta/creation_operators.h`

**Step 1: Write the template**

```cpp
#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <torch/library.h>
#include "../core/creation_common.h"

namespace torchscience::meta {

// MetaCreationTraits only needs output_shape - no kernel dispatch
template<typename CreationTraits>
struct MetaCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "meta_creation_op");

        auto options = at::TensorOptions()
            .dtype(dtype.value_or(
                c10::typeMetaToScalarType(at::get_default_dtype())
            ))
            .layout(layout.value_or(at::kStrided))
            .device(at::kMeta)
            .requires_grad(requires_grad);

        return at::empty(shape_vec, options);
    }
};

#define REGISTER_META_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::meta::MetaCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::meta
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/meta/creation_operators.h
git commit -m "$(cat <<'EOF'
feat: add MetaCreationOperator template

Template for shape inference on Meta device.
No kernel dispatch needed - only shape computation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Create Autocast creation operator template

**Files:**
- Create: `src/torchscience/csrc/autocast/creation_operators.h`

**Step 1: Write the template**

```cpp
#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast {

template<typename CreationTraits>
struct AutocastCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        c10::impl::ExcludeDispatchKeyGuard exclude_autocast(
            c10::DispatchKey::Autocast
        );

        // If dtype not specified, use autocast dtype based on device
        c10::optional<at::ScalarType> effective_dtype = dtype;
        if (!dtype.has_value()) {
            auto dev = device.value_or(at::kCPU);
            if (dev.is_cuda()) {
                effective_dtype = at::autocast::get_autocast_dtype(at::kCUDA);
            } else {
                effective_dtype = at::autocast::get_autocast_dtype(at::kCPU);
            }
        }

        return CreationTraits::dispatch_to_backend(
            args...,
            effective_dtype,
            layout,
            device,
            requires_grad
        );
    }
};

#define REGISTER_AUTOCAST_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::autocast::AutocastCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::autocast
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/autocast/creation_operators.h
git commit -m "$(cat <<'EOF'
feat: add AutocastCreationOperator template

Template for mixed precision casting of creation operators.
Determines effective dtype and redispatches to backend.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Create CPU stochastic creation operator template

**Files:**
- Create: `src/torchscience/csrc/cpu/stochastic_creation_operators.h`

**Step 1: Write the template**

```cpp
#pragma once

#include <vector>
#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/CPUGeneratorImpl.h>
#include <torch/library.h>
#include "../core/creation_common.h"

namespace torchscience::cpu {

// StochasticCreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t, typename RNG>
//     static void kernel(scalar_t* out, int64_t numel, RNG* rng, params...);

template<typename CreationTraits>
struct CPUStochasticCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        c10::optional<at::Generator> generator,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "stochastic_creation_op");

        auto options = ::torchscience::core::build_options(dtype, layout, device, at::kCPU);
        int64_t numel = ::torchscience::core::compute_numel(shape_vec);

        at::Tensor output = at::empty(shape_vec, options);

        if (numel > 0) {
            auto gen = at::get_generator_or_default<at::CPUGeneratorImpl>(
                generator, at::detail::getDefaultCPUGenerator()
            );
            std::lock_guard<std::mutex> lock(gen->mutex_);

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16,
                at::kHalf,
                output.scalar_type(),
                "cpu_stochastic_creation",
                [&]() {
                    CreationTraits::template kernel<scalar_t>(
                        output.data_ptr<scalar_t>(),
                        numel,
                        gen,
                        args...
                    );
                }
            );
        }

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_CPU_STOCHASTIC_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::cpu::CPUStochasticCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::cpu
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/stochastic_creation_operators.h
git commit -m "$(cat <<'EOF'
feat: add CPUStochasticCreationOperator template

Template for CPU creation operators with RNG support.
Handles generator locking and default generator fallback.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Create CUDA stochastic creation operator template

**Files:**
- Create: `src/torchscience/csrc/cuda/stochastic_creation_operators.cuh`

**Step 1: Write the template**

```cpp
#pragma once

#include <vector>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>
#include <torch/library.h>
#include "../core/creation_common.h"

namespace torchscience::cuda {

// CUDAStochasticCreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t>
//     static void launch_kernel(scalar_t* out, int64_t numel, at::PhiloxCudaState philox, params...);

template<typename CreationTraits>
struct CUDAStochasticCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        c10::optional<at::Generator> generator,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        auto dev = device.value_or(at::kCUDA);
        TORCH_CHECK(dev.is_cuda(), "device must be CUDA");

        c10::cuda::CUDAGuard guard(dev);

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "cuda_stochastic_creation_op");

        auto options = ::torchscience::core::build_options(dtype, layout, device, dev);
        int64_t numel = ::torchscience::core::compute_numel(shape_vec);

        at::Tensor output = at::empty(shape_vec, options);

        if (numel > 0) {
            auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
                generator, at::cuda::detail::getDefaultCUDAGenerator()
            );

            at::PhiloxCudaState philox_args;
            {
                std::lock_guard<std::mutex> lock(gen->mutex_);
                philox_args = gen->philox_cuda_state(numel);
            }

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16,
                at::kHalf,
                output.scalar_type(),
                "cuda_stochastic_creation",
                [&]() {
                    CreationTraits::template launch_kernel<scalar_t>(
                        output.data_ptr<scalar_t>(),
                        numel,
                        philox_args,
                        args...
                    );
                }
            );
        }

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_CUDA_STOCHASTIC_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::cuda::CUDAStochasticCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::cuda
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cuda/stochastic_creation_operators.cuh
git commit -m "$(cat <<'EOF'
feat: add CUDAStochasticCreationOperator template

Template for CUDA creation operators with Philox RNG.
Handles device guard and generator state.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Create Sparse COO CPU creation operator template

**Files:**
- Create: `src/torchscience/csrc/sparse/coo/cpu/creation_operators.h`

**Step 1: Write the template**

```cpp
#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../../core/creation_common.h"

namespace torchscience::sparse::coo::cpu {

// SparseCOOCreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t>
//     static std::pair<at::Tensor, at::Tensor> kernel(int64_t* shape, int64_t ndim, params...);
//     Returns (indices, values) tensors

template<typename CreationTraits>
struct SparseCOOCPUCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        (void)layout;  // Sparse layout is implicit

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "sparse_coo_cpu_creation_op");

        auto options = at::TensorOptions()
            .dtype(dtype.value_or(
                c10::typeMetaToScalarType(at::get_default_dtype())
            ))
            .device(device.value_or(at::kCPU))
            .requires_grad(false);

        at::Tensor indices;
        at::Tensor values;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16,
            at::kHalf,
            options.dtype().toScalarType(),
            "sparse_coo_cpu_creation",
            [&]() {
                auto result = CreationTraits::template kernel<scalar_t>(
                    shape_vec.data(),
                    static_cast<int64_t>(shape_vec.size()),
                    args...
                );
                indices = result.first;
                values = result.second.to(options);
            }
        );

        at::Tensor output = at::_sparse_coo_tensor_unsafe(
            indices,
            values,
            shape_vec,
            options.layout(at::kSparse)
        );

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_SPARSE_COO_CPU_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::sparse::coo::cpu::SparseCOOCPUCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::sparse::coo::cpu
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/sparse/coo/cpu/creation_operators.h
git commit -m "$(cat <<'EOF'
feat: add SparseCOOCPUCreationOperator template

Template for sparse COO tensor creation on CPU.
Kernel returns indices and values separately.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Create Sparse COO CUDA creation operator template

**Files:**
- Create: `src/torchscience/csrc/sparse/coo/cuda/creation_operators.cuh`

**Step 1: Write the template**

```cpp
#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "../../../core/creation_common.h"

namespace torchscience::sparse::coo::cuda {

template<typename CreationTraits>
struct SparseCOOCUDACreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        (void)layout;

        auto dev = device.value_or(at::kCUDA);
        TORCH_CHECK(dev.is_cuda(), "device must be CUDA");

        c10::cuda::CUDAGuard guard(dev);

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "sparse_coo_cuda_creation_op");

        auto options = at::TensorOptions()
            .dtype(dtype.value_or(
                c10::typeMetaToScalarType(at::get_default_dtype())
            ))
            .device(dev)
            .requires_grad(false);

        at::Tensor indices;
        at::Tensor values;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16,
            at::kHalf,
            options.dtype().toScalarType(),
            "sparse_coo_cuda_creation",
            [&]() {
                auto result = CreationTraits::template kernel<scalar_t>(
                    shape_vec.data(),
                    static_cast<int64_t>(shape_vec.size()),
                    args...
                );
                indices = result.first.to(dev);
                values = result.second.to(options);
            }
        );

        at::Tensor output = at::_sparse_coo_tensor_unsafe(
            indices,
            values,
            shape_vec,
            options.layout(at::kSparse)
        );

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_SPARSE_COO_CUDA_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::sparse::coo::cuda::SparseCOOCUDACreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::sparse::coo::cuda
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/sparse/coo/cuda/creation_operators.cuh
git commit -m "$(cat <<'EOF'
feat: add SparseCOOCUDACreationOperator template

Template for sparse COO tensor creation on CUDA.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Create Sparse CSR CPU creation operator template

**Files:**
- Create: `src/torchscience/csrc/sparse/csr/cpu/creation_operators.h`

**Step 1: Write the template**

```cpp
#pragma once

#include <vector>
#include <tuple>
#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../../core/creation_common.h"

namespace torchscience::sparse::csr::cpu {

// SparseCSRCreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t>
//     static std::tuple<at::Tensor, at::Tensor, at::Tensor> kernel(int64_t* shape, int64_t ndim, params...);
//     Returns (crow_indices, col_indices, values) tensors

template<typename CreationTraits>
struct SparseCSRCPUCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        (void)layout;

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "sparse_csr_cpu_creation_op");

        auto options = at::TensorOptions()
            .dtype(dtype.value_or(
                c10::typeMetaToScalarType(at::get_default_dtype())
            ))
            .device(device.value_or(at::kCPU))
            .requires_grad(false);

        at::Tensor crow_indices;
        at::Tensor col_indices;
        at::Tensor values;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16,
            at::kHalf,
            options.dtype().toScalarType(),
            "sparse_csr_cpu_creation",
            [&]() {
                auto result = CreationTraits::template kernel<scalar_t>(
                    shape_vec.data(),
                    static_cast<int64_t>(shape_vec.size()),
                    args...
                );
                crow_indices = std::get<0>(result);
                col_indices = std::get<1>(result);
                values = std::get<2>(result).to(options);
            }
        );

        at::Tensor output = at::sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            shape_vec,
            options
        );

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_SPARSE_CSR_CPU_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::sparse::csr::cpu::SparseCSRCPUCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::sparse::csr::cpu
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/sparse/csr/cpu/creation_operators.h
git commit -m "$(cat <<'EOF'
feat: add SparseCSRCPUCreationOperator template

Template for sparse CSR tensor creation on CPU.
Kernel returns crow_indices, col_indices, and values.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Create Sparse CSR CUDA creation operator template

**Files:**
- Create: `src/torchscience/csrc/sparse/csr/cuda/creation_operators.cuh`

**Step 1: Write the template**

```cpp
#pragma once

#include <vector>
#include <tuple>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "../../../core/creation_common.h"

namespace torchscience::sparse::csr::cuda {

template<typename CreationTraits>
struct SparseCSRCUDACreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        (void)layout;

        auto dev = device.value_or(at::kCUDA);
        TORCH_CHECK(dev.is_cuda(), "device must be CUDA");

        c10::cuda::CUDAGuard guard(dev);

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "sparse_csr_cuda_creation_op");

        auto options = at::TensorOptions()
            .dtype(dtype.value_or(
                c10::typeMetaToScalarType(at::get_default_dtype())
            ))
            .device(dev)
            .requires_grad(false);

        at::Tensor crow_indices;
        at::Tensor col_indices;
        at::Tensor values;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16,
            at::kHalf,
            options.dtype().toScalarType(),
            "sparse_csr_cuda_creation",
            [&]() {
                auto result = CreationTraits::template kernel<scalar_t>(
                    shape_vec.data(),
                    static_cast<int64_t>(shape_vec.size()),
                    args...
                );
                crow_indices = std::get<0>(result).to(dev);
                col_indices = std::get<1>(result).to(dev);
                values = std::get<2>(result).to(options);
            }
        );

        at::Tensor output = at::sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            shape_vec,
            options
        );

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_SPARSE_CSR_CUDA_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::sparse::csr::cuda::SparseCSRCUDACreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::sparse::csr::cuda
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/sparse/csr/cuda/creation_operators.cuh
git commit -m "$(cat <<'EOF'
feat: add SparseCSRCUDACreationOperator template

Template for sparse CSR tensor creation on CUDA.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Create Quantized CPU creation operator template

**Files:**
- Create: `src/torchscience/csrc/quantized/cpu/creation_operators.h`

**Step 1: Write the template**

```cpp
#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../core/creation_common.h"

namespace torchscience::quantized::cpu {

// QuantizedCreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t> static void kernel(scalar_t* out, int64_t numel, params...);

template<typename CreationTraits>
struct QuantizedCPUCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        double scale,
        int64_t zero_point,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        // Determine quantized type from requested dtype
        auto base_dtype = dtype.value_or(at::kFloat);
        at::ScalarType qtype;
        if (base_dtype == at::kFloat || base_dtype == at::kQInt8) {
            qtype = at::kQInt8;
        } else if (base_dtype == at::kQUInt8) {
            qtype = at::kQUInt8;
        } else if (base_dtype == at::kQInt32) {
            qtype = at::kQInt32;
        } else {
            qtype = at::kQInt8;
        }

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "quantized_cpu_creation_op");
        int64_t numel = ::torchscience::core::compute_numel(shape_vec);

        auto float_options = at::TensorOptions()
            .dtype(at::kFloat)
            .layout(layout.value_or(at::kStrided))
            .device(device.value_or(at::kCPU));

        // Create float tensor first, then quantize
        at::Tensor float_output = at::empty(shape_vec, float_options);

        if (numel > 0) {
            CreationTraits::template kernel<float>(
                float_output.data_ptr<float>(),
                numel,
                args...
            );
        }

        at::Tensor output = at::quantize_per_tensor(
            float_output, scale, zero_point, qtype
        );

        if (requires_grad) {
            TORCH_WARN("requires_grad ignored for quantized tensor");
        }

        return output;
    }
};

#define REGISTER_QUANTIZED_CPU_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::quantized::cpu::QuantizedCPUCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::quantized::cpu
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/quantized/cpu/creation_operators.h
git commit -m "$(cat <<'EOF'
feat: add QuantizedCPUCreationOperator template

Template for quantized tensor creation on CPU.
Creates float tensor then quantizes with scale/zero_point.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Create Quantized CUDA creation operator template

**Files:**
- Create: `src/torchscience/csrc/quantized/cuda/creation_operators.cuh`

**Step 1: Write the template**

```cpp
#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "../../core/creation_common.h"

namespace torchscience::quantized::cuda {

template<typename CreationTraits>
struct QuantizedCUDACreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        double scale,
        int64_t zero_point,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        auto dev = device.value_or(at::kCUDA);
        TORCH_CHECK(dev.is_cuda(), "device must be CUDA");

        c10::cuda::CUDAGuard guard(dev);

        auto base_dtype = dtype.value_or(at::kFloat);
        at::ScalarType qtype;
        if (base_dtype == at::kFloat || base_dtype == at::kQInt8) {
            qtype = at::kQInt8;
        } else if (base_dtype == at::kQUInt8) {
            qtype = at::kQUInt8;
        } else if (base_dtype == at::kQInt32) {
            qtype = at::kQInt32;
        } else {
            qtype = at::kQInt8;
        }

        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "quantized_cuda_creation_op");
        int64_t numel = ::torchscience::core::compute_numel(shape_vec);

        auto float_options = at::TensorOptions()
            .dtype(at::kFloat)
            .layout(layout.value_or(at::kStrided))
            .device(dev);

        at::Tensor float_output = at::empty(shape_vec, float_options);

        if (numel > 0) {
            CreationTraits::template launch_kernel<float>(
                float_output.data_ptr<float>(),
                numel,
                args...
            );
        }

        at::Tensor output = at::quantize_per_tensor(
            float_output, scale, zero_point, qtype
        );

        if (requires_grad) {
            TORCH_WARN("requires_grad ignored for quantized tensor");
        }

        return output;
    }
};

#define REGISTER_QUANTIZED_CUDA_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::quantized::cuda::QuantizedCUDACreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::quantized::cuda
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/quantized/cuda/creation_operators.cuh
git commit -m "$(cat <<'EOF'
feat: add QuantizedCUDACreationOperator template

Template for quantized tensor creation on CUDA.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Create example CreationTraits for rectangular_window

**Files:**
- Create: `src/torchscience/csrc/impl/window_function/rectangular_window_traits.h`

**Step 1: Write the traits class**

```cpp
#pragma once

#include <vector>

namespace torchscience::impl::window_function {

struct RectangularWindowTraits {
    static std::vector<int64_t> output_shape(int64_t n) {
        return {n};
    }

    template<typename scalar_t>
    static void kernel(scalar_t* output, int64_t numel, int64_t n) {
        (void)n;  // n == numel for rectangular window
        for (int64_t i = 0; i < numel; ++i) {
            output[i] = scalar_t(1);
        }
    }
};

}  // namespace torchscience::impl::window_function
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/impl/window_function/rectangular_window_traits.h
git commit -m "$(cat <<'EOF'
feat: add RectangularWindowTraits

Example CreationTraits for rectangular window function.
Demonstrates traits pattern for template-based operators.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Create example CreationTraits for sine_wave

**Files:**
- Create: `src/torchscience/csrc/impl/waveform/sine_wave_traits.h`

**Step 1: Write the traits class**

```cpp
#pragma once

#include <cmath>
#include <vector>

namespace torchscience::impl::waveform {

struct SineWaveTraits {
    static std::vector<int64_t> output_shape(
        int64_t n,
        double frequency,
        double sample_rate,
        double amplitude,
        double phase
    ) {
        (void)frequency;
        (void)sample_rate;
        (void)amplitude;
        (void)phase;
        return {n};
    }

    template<typename scalar_t>
    static void kernel(
        scalar_t* output,
        int64_t numel,
        int64_t n,
        double frequency,
        double sample_rate,
        double amplitude,
        double phase
    ) {
        (void)n;  // n == numel
        double angular_freq = 2.0 * M_PI * frequency / sample_rate;
        for (int64_t i = 0; i < numel; ++i) {
            output[i] = static_cast<scalar_t>(
                amplitude * std::sin(angular_freq * static_cast<double>(i) + phase)
            );
        }
    }
};

}  // namespace torchscience::impl::waveform
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/impl/waveform/sine_wave_traits.h
git commit -m "$(cat <<'EOF'
feat: add SineWaveTraits

Example CreationTraits for sine wave generation.
Shows multi-parameter traits pattern.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Migrate one operator (rectangular_window) as proof of concept

**Files:**
- Modify: Registration file (find the current rectangular_window registration)

**Step 1: Find current rectangular_window usage**

Run: `grep -r "rectangular_window" src/torchscience/csrc/ --include="*.h" --include="*.cpp" | head -20`
Expected: Locations of current rectangular_window macro usage

**Step 2: Update registration to use new template**

Replace the macro-based registration with template-based:

```cpp
#include "cpu/creation_operators.h"
#include "meta/creation_operators.h"
#include "impl/window_function/rectangular_window_traits.h"

using torchscience::impl::window_function::RectangularWindowTraits;

// In CPU TORCH_LIBRARY_IMPL block:
REGISTER_CPU_CREATION(module, rectangular_window, RectangularWindowTraits, int64_t);

// In Meta TORCH_LIBRARY_IMPL block:
REGISTER_META_CREATION(module, rectangular_window, RectangularWindowTraits, int64_t);
```

**Step 3: Verify tests pass**

Run: `uv run pytest tests/torchscience/signal_processing/window_function/test__rectangular_window.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat: migrate rectangular_window to template-based registration

Proof of concept migration using CPUCreationOperator and
MetaCreationOperator templates with RectangularWindowTraits.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Verify rectangular_window works end-to-end

**Step 1: Run verification script**

Run:
```bash
python -c "
import torch
import torchscience

# Test basic creation
w = torchscience.rectangular_window(10)
print(f'Shape: {w.shape}')
print(f'Values: {w}')
assert w.shape == (10,)
assert torch.all(w == 1.0)
print('Basic creation: OK')

# Test dtype
w_f64 = torchscience.rectangular_window(5, dtype=torch.float64)
assert w_f64.dtype == torch.float64
print('Dtype handling: OK')

# Test empty
w_empty = torchscience.rectangular_window(0)
assert w_empty.shape == (0,)
print('Empty tensor: OK')

# Test Meta device
w_meta = torchscience.rectangular_window(10, device='meta')
assert w_meta.device.type == 'meta'
assert w_meta.shape == (10,)
print('Meta device: OK')

print('All verification tests passed!')
"
```

Expected: All checks pass

**Step 2: (No commit needed - verification only)**

---

### Task 17: Document migration pattern for remaining operators

**Files:**
- Create: `docs/migration-guide-creation-operators.md`

**Step 1: Write migration guide**

```markdown
# Migration Guide: Macro to Template-Based Creation Operators

## Overview

This guide explains how to migrate from `CPU_CREATION_OPERATOR` macros to
`CPUCreationOperator<Traits>` templates.

## Step 1: Create a Traits Class

For each operator, create a traits class in `impl/<namespace>/<operator>_traits.h`:

```cpp
struct MyOperatorTraits {
    static std::vector<int64_t> output_shape(/* params */) {
        return {/* shape */};
    }

    template<typename scalar_t>
    static void kernel(scalar_t* output, int64_t numel, /* params */) {
        // Fill output
    }
};
```

## Step 2: Register Using Template

Replace macro with template registration:

```cpp
// Before (macro)
CPU_CREATION_OPERATOR(namespace, my_op, {n}, (int64_t n), (n))

// After (template)
#include "cpu/creation_operators.h"
#include "impl/namespace/my_op_traits.h"

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    REGISTER_CPU_CREATION(module, my_op, MyOpTraits, int64_t);
}
```

## Step 3: Run Tests

Verify existing tests still pass after migration.
```

**Step 2: Commit**

```bash
git add docs/migration-guide-creation-operators.md
git commit -m "$(cat <<'EOF'
docs: add migration guide for template-based creation operators

Explains step-by-step how to migrate from macros to templates.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## File Structure After Initial Implementation

```
src/torchscience/csrc/
├── core/
│   └── creation_common.h              # Existing shared utilities
├── cpu/
│   ├── creation_operators.h           # NEW: CPUCreationOperator template
│   ├── stochastic_creation_operators.h # NEW: CPUStochasticCreationOperator
│   └── creation_macros.h              # LEGACY: To be removed after full migration
├── cuda/
│   ├── creation_operators.cuh         # NEW: CUDACreationOperator template
│   ├── stochastic_creation_operators.cuh # NEW
│   └── creation_macros.cuh            # LEGACY
├── meta/
│   ├── creation_operators.h           # NEW: MetaCreationOperator template
│   └── creation_macros.h              # LEGACY
├── autocast/
│   ├── creation_operators.h           # NEW: AutocastCreationOperator
│   └── creation_macros.h              # LEGACY
├── sparse/
│   ├── coo/
│   │   ├── cpu/creation_operators.h   # NEW
│   │   └── cuda/creation_operators.cuh # NEW
│   └── csr/
│       ├── cpu/creation_operators.h   # NEW
│       └── cuda/creation_operators.cuh # NEW
├── quantized/
│   ├── cpu/creation_operators.h       # NEW
│   └── cuda/creation_operators.cuh    # NEW
└── impl/
    ├── window_function/
    │   └── rectangular_window_traits.h # NEW: Example traits
    └── waveform/
        └── sine_wave_traits.h         # NEW: Example traits
```

---

## Benefits

1. **Type Safety:** Template logic is type-checked at compile time
2. **Debuggability:** Step through template code in debugger (no macro expansion)
3. **IDE Support:** Autocomplete, go-to-definition work on templates
4. **Reduced Duplication:** ~1,100 lines -> ~600 lines of templates + ~12 lines of macros
5. **Clear Separation:** Templates = logic, Macros = only string concatenation
6. **Reuses Infrastructure:** Leverages existing `core/creation_common.h`

## Macro Comparison

| Before | After |
|--------|-------|
| `CPU_CREATION_OPERATOR(ns, name, shape, params, args)` (~75 lines) | `REGISTER_CPU_CREATION(m, name, Traits, ...)` (1 line) |
| Macro contains all logic | Macro only does `#name` |
| Hard to debug | Templates are debuggable |
| No type checking | Full type checking |

---

## Verification Checklist

- [ ] CPUCreationOperator compiles
- [ ] CUDACreationOperator compiles
- [ ] MetaCreationOperator compiles
- [ ] AutocastCreationOperator compiles
- [ ] CPUStochasticCreationOperator compiles
- [ ] CUDAStochasticCreationOperator compiles
- [ ] SparseCOOCPUCreationOperator compiles
- [ ] SparseCOOCUDACreationOperator compiles
- [ ] SparseCSRCPUCreationOperator compiles
- [ ] SparseCSRCUDACreationOperator compiles
- [ ] QuantizedCPUCreationOperator compiles
- [ ] QuantizedCUDACreationOperator compiles
- [ ] rectangular_window passes all existing tests
- [ ] Migration guide documented
