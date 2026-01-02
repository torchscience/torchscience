# Window Function Module Design

**Date:** 2026-01-02
**Status:** Approved
**Module:** `torchscience.signal_processing.window_function`

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API style | Separate functions (`hann_window` + `periodic_hann_window`) | Clearer intent, explicit behavior |
| Window length | `Union[int, Tensor]` n | Enables batched variable-length windows |
| Variable-length output | NestedTensor | Native PyTorch support, no padding waste |
| Frequency properties | Skip | Users compute if needed, reduces API surface |
| Coeffs batching | `(K,)` or `(B, K)` supported | Flexibility for batch-specific coefficients |
| C++ pattern | New minimal-boilerplate pattern | ~12 lines per window, performance-focused |
| Window unification | Same code path for all | Consistent infrastructure, easier maintenance |

## Overview

Comprehensive window function module providing ~44 functions across 3 phases with:

- **Differentiable parameters** — learn optimal window shapes via autograd
- **Batched generation** — efficient parallel window creation with different parameters
- **Variable-length batching** — Tensor `n` with NestedTensor output for variable window lengths
- **Unified API** — consistent signatures, explicit symmetric/periodic variants
- **C++ backend** — new minimal-boilerplate pattern optimized for performance

## Value Proposition

| Feature | torch.signal.windows | scipy.signal.windows | torchscience |
|---------|---------------------|---------------------|--------------|
| Window count | 11 | 26 | 44 (planned) |
| Differentiable params | No | No | Yes |
| Batched generation | No | No | Yes |
| Variable-length batch | No | No | Yes (NestedTensor) |
| Unified API | Yes | Yes | Yes |
| GPU support | Yes | No | Yes |

## API Design

### Naming Convention

Each window has two variants:

- **Symmetric:** `{name}_window(n, ...)` — for filter design, FIR coefficients
- **Periodic:** `periodic_{name}_window(n, ...)` — for spectral analysis, STFT

### Signature Patterns

**Parameterless windows:**

```python
def hann_window(
    n: Union[int, Tensor],  # scalar int, 0-D tensor, or 1-D tensor of lengths
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Union[Tensor, NestedTensor]:
    """
    Returns:
    - n is int or 0-D: shape (n,)
    - n is 1-D tensor [n1, n2, ...]: NestedTensor with shapes [(n1,), (n2,), ...]
    """
```

**Parameterized windows:**

```python
def gaussian_window(
    n: Union[int, Tensor],  # scalar or batched lengths
    std: Tensor,            # shape () or (B,) - differentiable, batchable
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Union[Tensor, NestedTensor]:
    """
    n: window length(s) - int, 0-D tensor, or 1-D tensor of lengths
    std: standard deviation parameter (differentiable, batchable)

    When n is batched (1-D tensor), std must broadcast with n.shape[0].
    requires_grad inherited from std tensor.

    Returns NestedTensor when n is 1-D tensor with variable lengths.
    """
```

### Batching Semantics

**Window length batching (new):**

| `n` type | Output type | Example |
|----------|-------------|---------|
| `int` or `tensor(5)` | `Tensor` shape `(n,)` | `hann_window(5)` → `(5,)` |
| `tensor([5, 8, 3])` | `NestedTensor` | 3 windows of lengths 5, 8, 3 |

**Parameter batching:**

| `std` shape | `n` type | Output |
|-------------|----------|--------|
| `()` scalar | `int` | `(n,)` |
| `(B,)` | `int` | `(B, n)` — all same length |
| `()` scalar | `(B,)` tensor | NestedTensor with B windows |
| `(B,)` | `(B,)` tensor | NestedTensor with B windows, each with its own std |

**Broadcasting rules:**
- When both `n` and parameter are batched, their batch dimensions must broadcast
- Parameter batching produces regular `Tensor` with fixed `(B, n)` shape
- Length batching produces `NestedTensor` with variable shapes

## Phase 1: PyTorch Parity (18 functions)

| Window | Parameters | Symmetric | Periodic |
|--------|------------|-----------|----------|
| Bartlett | none | `bartlett_window` | `periodic_bartlett_window` |
| Blackman | none | `blackman_window` | `periodic_blackman_window` |
| Cosine | none | `cosine_window` | `periodic_cosine_window` |
| Gaussian | `std: Tensor` | `gaussian_window` | `periodic_gaussian_window` |
| Hamming | none | `hamming_window` | `periodic_hamming_window` |
| Hann | none | `hann_window` | `periodic_hann_window` |
| Nuttall | none | `nuttall_window` | `periodic_nuttall_window` |
| General Cosine | `coeffs: Tensor` | `general_cosine_window` | `periodic_general_cosine_window` |
| General Hamming | `alpha: Tensor` | `general_hamming_window` | `periodic_general_hamming_window` |

## Phase 2: Extended Coverage (16 functions)

| Window | Parameters |
|--------|------------|
| Kaiser | `beta: Tensor` |
| Tukey | `alpha: Tensor` |
| Blackman-Harris | none |
| Flat Top | none |
| Bohman | none |
| Parzen | none |
| Lanczos | none |
| Triangular | none |

## Phase 3: Specialized Windows (10 functions)

| Window | Parameters | Notes |
|--------|------------|-------|
| Chebyshev | `attenuation: Tensor` | Equiripple sidelobes |
| DPSS | `bandwidth: Tensor` | Slepian sequences |
| Exponential | `tau: Tensor` | Decay parameter |
| Taylor | `nbar, sll: Tensor` | Radar applications |
| Kaiser-Bessel Derived | `beta: Tensor` | MDCT applications |

## C++ Architecture

### Design Goals

1. **Minimal boilerplate** — Adding a new window should require ~15 lines total
2. **Performance** — SIMD-friendly, parallel-ready, no abstraction overhead
3. **Unified path** — Same infrastructure for parameterless and parameterized windows
4. **Macro-based** — Similar pattern to pointwise operators but adapted for creation ops
5. **NestedTensor support** — First-class variable-length batch output

### File Structure

Following the same pattern as `cpu/special_functions.h`, `meta/special_functions.h`, etc.:

```
src/torchscience/csrc/
├── cpu/
│   ├── window_functions.h        # CPU implementations + TORCH_LIBRARY_IMPL
│   └── macros/
│       └── window_function.h     # CPU macro definitions
├── meta/
│   ├── window_functions.h        # Meta implementations + TORCH_LIBRARY_IMPL
│   └── macros/
│       └── window_function.h     # Meta macro definitions
├── autograd/
│   ├── window_functions.h        # Autograd implementations + TORCH_LIBRARY_IMPL
│   └── macros/
│       └── window_function.h     # Autograd macro definitions
└── kernel/signal_processing/
    └── window_function/
        ├── common.h              # Shared utilities (window_denominator)
        ├── hann.h                # Hann kernel math
        ├── gaussian.h            # Gaussian kernel math + derivatives
        └── ...
```

### Macro Design Philosophy

Unlike pointwise operators which iterate over tensor elements, window functions:
1. Take a scalar length `n` (not a tensor of values)
2. Generate a new tensor of shape `(n,)` or `(B, n)` or NestedTensor
3. Only parameterized windows have gradients (w.r.t. parameters, not `n`)

This leads to two distinct macro categories:
- **Parameterless**: No backward pass needed (Hann, Hamming, Blackman, etc.)
- **Parameterized**: Backward pass for parameter gradients (Gaussian, Kaiser, etc.)

### CPU Macros

```cpp
// cpu/macros/window_function.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

// =============================================================================
// Parameterless window: no backward pass needed
// Usage: TORCHSCIENCE_CPU_WINDOW_PARAMETERLESS(hann)
// Expects kernel::window_function::hann<scalar_t>(i, n, periodic) to exist
// =============================================================================

#define TORCHSCIENCE_CPU_WINDOW_PARAMETERLESS(name)                            \
namespace torchscience::cpu::window_function {                                  \
                                                                                \
inline at::Tensor name##_window(                                                \
  int64_t n,                                                                    \
  bool periodic,                                                                \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  TORCH_CHECK(n >= 0, #name "_window: n must be non-negative, got ", n);        \
                                                                                \
  auto options = at::TensorOptions()                                            \
    .dtype(dtype.value_or(at::get_default_dtype_as_scalartype()))               \
    .layout(layout.value_or(at::kStrided))                                      \
    .device(device.value_or(at::kCPU));                                         \
                                                                                \
  auto output = at::empty({n}, options);                                        \
                                                                                \
  if (n == 0) {                                                                 \
    return output.requires_grad_(requires_grad);                                \
  }                                                                             \
                                                                                \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                              \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    output.scalar_type(),                                                       \
    #name "_window",                                                            \
    [&] {                                                                       \
      auto* out_ptr = output.data_ptr<scalar_t>();                              \
      at::parallel_for(0, n, /*grain_size=*/1024, [&](int64_t begin, int64_t end) { \
        for (int64_t i = begin; i < end; ++i) {                                 \
          out_ptr[i] = kernel::window_function::name<scalar_t>(i, n, periodic); \
        }                                                                       \
      });                                                                       \
    }                                                                           \
  );                                                                            \
                                                                                \
  return output.requires_grad_(requires_grad);                                  \
}                                                                               \
                                                                                \
} /* namespace torchscience::cpu::window_function */                            \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                                 \
  module.impl(                                                                  \
    #name "_window",                                                            \
    [](int64_t n,                                                               \
       c10::optional<at::ScalarType> dtype,                                     \
       c10::optional<at::Layout> layout,                                        \
       c10::optional<at::Device> device,                                        \
       bool requires_grad) {                                                    \
      return torchscience::cpu::window_function::name##_window(                 \
        n, /*periodic=*/false, dtype, layout, device, requires_grad);           \
    }                                                                           \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    "periodic_" #name "_window",                                                \
    [](int64_t n,                                                               \
       c10::optional<at::ScalarType> dtype,                                     \
       c10::optional<at::Layout> layout,                                        \
       c10::optional<at::Device> device,                                        \
       bool requires_grad) {                                                    \
      return torchscience::cpu::window_function::name##_window(                 \
        n, /*periodic=*/true, dtype, layout, device, requires_grad);            \
    }                                                                           \
  );                                                                            \
}

// =============================================================================
// Parameterized window with 1 parameter: has backward pass
// Usage: TORCHSCIENCE_CPU_WINDOW_1PARAM(gaussian, std)
// Expects:
//   kernel::window_function::gaussian<scalar_t>(i, n, std, periodic)
//   kernel::window_function::gaussian_backward<scalar_t>(grad_out, i, n, std, periodic, fwd_value)
// =============================================================================

#define TORCHSCIENCE_CPU_WINDOW_1PARAM(name, param)                             \
namespace torchscience::cpu::window_function {                                  \
                                                                                \
inline at::Tensor name##_window(                                                \
  int64_t n,                                                                    \
  const at::Tensor& param##_input,                                              \
  bool periodic,                                                                \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device                                              \
) {                                                                             \
  TORCH_CHECK(n >= 0, #name "_window: n must be non-negative, got ", n);        \
                                                                                \
  /* Determine output dtype from param if not specified */                      \
  auto out_dtype = dtype.value_or(param##_input.scalar_type());                 \
  auto options = at::TensorOptions()                                            \
    .dtype(out_dtype)                                                           \
    .layout(layout.value_or(at::kStrided))                                      \
    .device(device.value_or(param##_input.device()));                           \
                                                                                \
  /* Handle batched parameters: param shape () -> (n,), shape (B,) -> (B, n) */ \
  int64_t batch = param##_input.dim() == 0 ? 0 : param##_input.size(0);         \
                                                                                \
  at::Tensor output;                                                            \
  if (batch == 0) {                                                             \
    output = at::empty({n}, options);                                           \
  } else {                                                                      \
    output = at::empty({batch, n}, options);                                    \
  }                                                                             \
                                                                                \
  if (n == 0) return output;                                                    \
                                                                                \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                              \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    output.scalar_type(),                                                       \
    #name "_window",                                                            \
    [&] {                                                                       \
      if (batch == 0) {                                                         \
        auto* out_ptr = output.data_ptr<scalar_t>();                            \
        scalar_t param##_val = param##_input.item<scalar_t>();                  \
        at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {          \
          for (int64_t i = begin; i < end; ++i) {                               \
            out_ptr[i] = kernel::window_function::name<scalar_t>(               \
              i, n, param##_val, periodic);                                     \
          }                                                                     \
        });                                                                     \
      } else {                                                                  \
        auto out_a = output.accessor<scalar_t, 2>();                            \
        auto param##_a = param##_input.accessor<scalar_t, 1>();                 \
        at::parallel_for(0, batch * n, 1024, [&](int64_t begin, int64_t end) {  \
          for (int64_t idx = begin; idx < end; ++idx) {                         \
            int64_t b = idx / n;                                                \
            int64_t i = idx % n;                                                \
            out_a[b][i] = kernel::window_function::name<scalar_t>(              \
              i, n, param##_a[b], periodic);                                    \
          }                                                                     \
        });                                                                     \
      }                                                                         \
    }                                                                           \
  );                                                                            \
                                                                                \
  return output;                                                                \
}                                                                               \
                                                                                \
inline at::Tensor name##_window_backward(                                       \
  const at::Tensor& grad_output,                                                \
  const at::Tensor& output,                                                     \
  int64_t n,                                                                    \
  const at::Tensor& param##_input,                                              \
  bool periodic                                                                 \
) {                                                                             \
  /* Gradient w.r.t. param: sum over window indices */                          \
  int64_t batch = param##_input.dim() == 0 ? 0 : param##_input.size(0);         \
                                                                                \
  at::Tensor grad_param;                                                        \
  if (batch == 0) {                                                             \
    grad_param = at::zeros_like(param##_input);                                 \
  } else {                                                                      \
    grad_param = at::zeros({batch}, param##_input.options());                   \
  }                                                                             \
                                                                                \
  if (n == 0) return grad_param;                                                \
                                                                                \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                              \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    grad_output.scalar_type(),                                                  \
    #name "_window_backward",                                                   \
    [&] {                                                                       \
      if (batch == 0) {                                                         \
        auto* grad_out_ptr = grad_output.data_ptr<scalar_t>();                  \
        auto* out_ptr = output.data_ptr<scalar_t>();                            \
        scalar_t param##_val = param##_input.item<scalar_t>();                  \
        scalar_t accum = 0;                                                     \
        for (int64_t i = 0; i < n; ++i) {                                       \
          accum += kernel::window_function::name##_backward<scalar_t>(          \
            grad_out_ptr[i], i, n, param##_val, periodic, out_ptr[i]);          \
        }                                                                       \
        grad_param.fill_(accum);                                                \
      } else {                                                                  \
        auto grad_out_a = grad_output.accessor<scalar_t, 2>();                  \
        auto out_a = output.accessor<scalar_t, 2>();                            \
        auto param##_a = param##_input.accessor<scalar_t, 1>();                 \
        auto grad_param##_a = grad_param.accessor<scalar_t, 1>();               \
        for (int64_t b = 0; b < batch; ++b) {                                   \
          scalar_t accum = 0;                                                   \
          for (int64_t i = 0; i < n; ++i) {                                     \
            accum += kernel::window_function::name##_backward<scalar_t>(        \
              grad_out_a[b][i], i, n, param##_a[b], periodic, out_a[b][i]);     \
          }                                                                     \
          grad_param##_a[b] = accum;                                            \
        }                                                                       \
      }                                                                         \
    }                                                                           \
  );                                                                            \
                                                                                \
  return grad_param;                                                            \
}                                                                               \
                                                                                \
} /* namespace torchscience::cpu::window_function */                            \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                                 \
  module.impl(                                                                  \
    #name "_window",                                                            \
    [](int64_t n,                                                               \
       const at::Tensor& param,                                                 \
       c10::optional<at::ScalarType> dtype,                                     \
       c10::optional<at::Layout> layout,                                        \
       c10::optional<at::Device> device) {                                      \
      return torchscience::cpu::window_function::name##_window(                 \
        n, param, /*periodic=*/false, dtype, layout, device);                   \
    }                                                                           \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    "periodic_" #name "_window",                                                \
    [](int64_t n,                                                               \
       const at::Tensor& param,                                                 \
       c10::optional<at::ScalarType> dtype,                                     \
       c10::optional<at::Layout> layout,                                        \
       c10::optional<at::Device> device) {                                      \
      return torchscience::cpu::window_function::name##_window(                 \
        n, param, /*periodic=*/true, dtype, layout, device);                    \
    }                                                                           \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    #name "_window_backward",                                                   \
    torchscience::cpu::window_function::name##_window_backward                  \
  );                                                                            \
                                                                                \
  module.impl(                                                                  \
    "periodic_" #name "_window_backward",                                       \
    [](const at::Tensor& grad_output,                                           \
       const at::Tensor& output,                                                \
       int64_t n,                                                               \
       const at::Tensor& param) {                                               \
      return torchscience::cpu::window_function::name##_window_backward(        \
        grad_output, output, n, param, /*periodic=*/true);                      \
    }                                                                           \
  );                                                                            \
}
```

### Meta Macros

```cpp
// meta/macros/window_function.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#define TORCHSCIENCE_META_WINDOW_PARAMETERLESS(name)                            \
namespace torchscience::meta::window_function {                                 \
                                                                                \
inline at::Tensor name##_window(                                                \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  auto options = at::TensorOptions()                                            \
    .dtype(dtype.value_or(at::get_default_dtype_as_scalartype()))               \
    .layout(layout.value_or(at::kStrided))                                      \
    .device(at::kMeta);                                                         \
  return at::empty({n}, options);                                               \
}                                                                               \
                                                                                \
} /* namespace torchscience::meta::window_function */                           \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                                \
  module.impl(#name "_window", torchscience::meta::window_function::name##_window); \
  module.impl("periodic_" #name "_window", torchscience::meta::window_function::name##_window); \
}

#define TORCHSCIENCE_META_WINDOW_1PARAM(name, param)                            \
namespace torchscience::meta::window_function {                                 \
                                                                                \
inline at::Tensor name##_window(                                                \
  int64_t n,                                                                    \
  const at::Tensor& param##_input,                                              \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device                                              \
) {                                                                             \
  auto out_dtype = dtype.value_or(param##_input.scalar_type());                 \
  auto options = at::TensorOptions()                                            \
    .dtype(out_dtype)                                                           \
    .layout(layout.value_or(at::kStrided))                                      \
    .device(at::kMeta);                                                         \
                                                                                \
  int64_t batch = param##_input.dim() == 0 ? 0 : param##_input.size(0);         \
  if (batch == 0) {                                                             \
    return at::empty({n}, options);                                             \
  } else {                                                                      \
    return at::empty({batch, n}, options);                                      \
  }                                                                             \
}                                                                               \
                                                                                \
inline at::Tensor name##_window_backward(                                       \
  const at::Tensor& grad_output,                                                \
  const at::Tensor& output,                                                     \
  int64_t n,                                                                    \
  const at::Tensor& param##_input,                                              \
  bool periodic                                                                 \
) {                                                                             \
  return at::empty_like(param##_input, at::TensorOptions().device(at::kMeta));  \
}                                                                               \
                                                                                \
} /* namespace torchscience::meta::window_function */                           \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                                \
  module.impl(                                                                  \
    #name "_window",                                                            \
    torchscience::meta::window_function::name##_window                          \
  );                                                                            \
  module.impl(                                                                  \
    "periodic_" #name "_window",                                                \
    torchscience::meta::window_function::name##_window                          \
  );                                                                            \
  module.impl(                                                                  \
    #name "_window_backward",                                                   \
    torchscience::meta::window_function::name##_window_backward                 \
  );                                                                            \
  module.impl(                                                                  \
    "periodic_" #name "_window_backward",                                       \
    torchscience::meta::window_function::name##_window_backward                 \
  );                                                                            \
}
```

### Autograd Macros

```cpp
// autograd/macros/window_function.h
#pragma once

#include <torch/extension.h>

// Parameterless windows don't need autograd wrappers - use CompositeExplicitAutograd

#define TORCHSCIENCE_AUTOGRAD_WINDOW_1PARAM(name, Name, param)                  \
namespace torchscience::autograd::window_function {                             \
                                                                                \
class Name##WindowBackward                                                      \
    : public torch::autograd::Function<Name##WindowBackward> {                  \
public:                                                                         \
  static std::vector<at::Tensor> forward(                                       \
    torch::autograd::AutogradContext* ctx,                                      \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& output,                                                   \
    int64_t n,                                                                  \
    const at::Tensor& param##_input,                                            \
    bool periodic,                                                              \
    bool param##_requires_grad                                                  \
  ) {                                                                           \
    ctx->save_for_backward({grad_output, output, param##_input});               \
    ctx->saved_data["n"] = n;                                                   \
    ctx->saved_data["periodic"] = periodic;                                     \
    ctx->saved_data[#param "_requires_grad"] = param##_requires_grad;           \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    std::string op_name = periodic                                              \
      ? "torchscience::periodic_" #name "_window_backward"                      \
      : "torchscience::" #name "_window_backward";                              \
                                                                                \
    auto grad_param = c10::Dispatcher::singleton()                              \
      .findSchemaOrThrow(op_name, "")                                           \
      .typed<at::Tensor(                                                        \
        const at::Tensor&, const at::Tensor&, int64_t,                          \
        const at::Tensor&, bool)>()                                             \
      .call(grad_output, output, n, param##_input, periodic);                   \
                                                                                \
    return {grad_param};                                                        \
  }                                                                             \
                                                                                \
  static std::vector<at::Tensor> backward(                                      \
    torch::autograd::AutogradContext* ctx,                                      \
    const std::vector<at::Tensor>& grad_outputs                                 \
  ) {                                                                           \
    /* Second-order gradients: typically not needed for window functions */     \
    return {at::Tensor(), at::Tensor(), at::Tensor(),                           \
            at::Tensor(), at::Tensor(), at::Tensor()};                          \
  }                                                                             \
};                                                                              \
                                                                                \
class Name##Window : public torch::autograd::Function<Name##Window> {           \
public:                                                                         \
  static at::Tensor forward(                                                    \
    torch::autograd::AutogradContext* ctx,                                      \
    int64_t n,                                                                  \
    const at::Tensor& param##_input,                                            \
    bool periodic,                                                              \
    c10::optional<at::ScalarType> dtype,                                        \
    c10::optional<at::Layout> layout,                                           \
    c10::optional<at::Device> device                                            \
  ) {                                                                           \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    std::string op_name = periodic                                              \
      ? "torchscience::periodic_" #name "_window"                               \
      : "torchscience::" #name "_window";                                       \
                                                                                \
    auto output = c10::Dispatcher::singleton()                                  \
      .findSchemaOrThrow(op_name, "")                                           \
      .typed<at::Tensor(                                                        \
        int64_t, const at::Tensor&,                                             \
        c10::optional<at::ScalarType>,                                          \
        c10::optional<at::Layout>,                                              \
        c10::optional<at::Device>)>()                                           \
      .call(n, param##_input, dtype, layout, device);                           \
                                                                                \
    ctx->save_for_backward({output, param##_input});                            \
    ctx->saved_data["n"] = n;                                                   \
    ctx->saved_data["periodic"] = periodic;                                     \
    ctx->saved_data[#param "_requires_grad"] =                                  \
      param##_input.requires_grad() &&                                          \
      at::isFloatingType(param##_input.scalar_type());                          \
                                                                                \
    return output;                                                              \
  }                                                                             \
                                                                                \
  static torch::autograd::variable_list backward(                               \
    torch::autograd::AutogradContext* ctx,                                      \
    const torch::autograd::variable_list& grad_outputs                          \
  ) {                                                                           \
    auto saved = ctx->get_saved_variables();                                    \
    int64_t n = ctx->saved_data["n"].toInt();                                   \
    bool periodic = ctx->saved_data["periodic"].toBool();                       \
    bool param##_requires_grad =                                                \
      ctx->saved_data[#param "_requires_grad"].toBool();                        \
                                                                                \
    at::Tensor grad_param;                                                      \
    if (param##_requires_grad && grad_outputs[0].defined()) {                   \
      auto grads = Name##WindowBackward::apply(                                 \
        grad_outputs[0], saved[0], n, saved[1], periodic, true);                \
      grad_param = grads[0];                                                    \
    }                                                                           \
                                                                                \
    return {at::Tensor(), grad_param, at::Tensor(),                             \
            at::Tensor(), at::Tensor(), at::Tensor()};                          \
  }                                                                             \
};                                                                              \
                                                                                \
inline at::Tensor name##_window(                                                \
  int64_t n,                                                                    \
  const at::Tensor& param,                                                      \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device                                              \
) {                                                                             \
  return Name##Window::apply(n, param, false, dtype, layout, device);           \
}                                                                               \
                                                                                \
inline at::Tensor periodic_##name##_window(                                     \
  int64_t n,                                                                    \
  const at::Tensor& param,                                                      \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device                                              \
) {                                                                             \
  return Name##Window::apply(n, param, true, dtype, layout, device);            \
}                                                                               \
                                                                                \
} /* namespace torchscience::autograd::window_function */                       \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                            \
  module.impl(#name "_window",                                                  \
    torchscience::autograd::window_function::name##_window);                    \
  module.impl("periodic_" #name "_window",                                      \
    torchscience::autograd::window_function::periodic_##name##_window);         \
}
```

### Usage Files

Following the same pattern as `cpu/special_functions.h`:

```cpp
// cpu/window_functions.h
#pragma once

#include "macros/window_function.h"

// Kernel includes
#include "../kernel/signal_processing/window_function/hann.h"
#include "../kernel/signal_processing/window_function/hamming.h"
#include "../kernel/signal_processing/window_function/blackman.h"
#include "../kernel/signal_processing/window_function/gaussian.h"
// ...

// Parameterless windows
TORCHSCIENCE_CPU_WINDOW_PARAMETERLESS(hann)
TORCHSCIENCE_CPU_WINDOW_PARAMETERLESS(hamming)
TORCHSCIENCE_CPU_WINDOW_PARAMETERLESS(blackman)
TORCHSCIENCE_CPU_WINDOW_PARAMETERLESS(bartlett)
TORCHSCIENCE_CPU_WINDOW_PARAMETERLESS(cosine)
TORCHSCIENCE_CPU_WINDOW_PARAMETERLESS(nuttall)

// Parameterized windows
TORCHSCIENCE_CPU_WINDOW_1PARAM(gaussian, std)
TORCHSCIENCE_CPU_WINDOW_1PARAM(kaiser, beta)
TORCHSCIENCE_CPU_WINDOW_1PARAM(general_hamming, alpha)
```

```cpp
// meta/window_functions.h
#pragma once

#include "macros/window_function.h"

TORCHSCIENCE_META_WINDOW_PARAMETERLESS(hann)
TORCHSCIENCE_META_WINDOW_PARAMETERLESS(hamming)
TORCHSCIENCE_META_WINDOW_PARAMETERLESS(blackman)
TORCHSCIENCE_META_WINDOW_PARAMETERLESS(bartlett)
TORCHSCIENCE_META_WINDOW_PARAMETERLESS(cosine)
TORCHSCIENCE_META_WINDOW_PARAMETERLESS(nuttall)

TORCHSCIENCE_META_WINDOW_1PARAM(gaussian, std)
TORCHSCIENCE_META_WINDOW_1PARAM(kaiser, beta)
TORCHSCIENCE_META_WINDOW_1PARAM(general_hamming, alpha)
```

```cpp
// autograd/window_functions.h
#pragma once

#include "macros/window_function.h"

// Only parameterized windows need autograd wrappers
TORCHSCIENCE_AUTOGRAD_WINDOW_1PARAM(gaussian, Gaussian, std)
TORCHSCIENCE_AUTOGRAD_WINDOW_1PARAM(kaiser, Kaiser, beta)
TORCHSCIENCE_AUTOGRAD_WINDOW_1PARAM(general_hamming, GeneralHamming, alpha)
```

### Kernel Interface

Each window kernel is a simple inline function:

```cpp
// kernel/signal_processing/window_function/common.h
#pragma once

#include <cmath>

namespace torchscience::kernel::window_function {

template<typename scalar_t>
inline scalar_t window_denominator(int64_t n, bool periodic) {
  // periodic: denominator is n (for FFT compatibility)
  // symmetric: denominator is n-1 (for filter design)
  return periodic ? static_cast<scalar_t>(n) : static_cast<scalar_t>(n - 1);
}

}  // namespace torchscience::kernel::window_function
```

```cpp
// kernel/signal_processing/window_function/hann.h
#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

template<typename scalar_t>
inline scalar_t hann(int64_t i, int64_t n, bool periodic) {
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == 0) return scalar_t(1);  // n=1 case
  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  return scalar_t(0.5) * (scalar_t(1) - std::cos(x));
}

}  // namespace torchscience::kernel::window_function
```

```cpp
// kernel/signal_processing/window_function/gaussian.h
#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

template<typename scalar_t>
inline scalar_t gaussian(int64_t i, int64_t n, scalar_t std, bool periodic) {
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  scalar_t center = denom / scalar_t(2);
  scalar_t x = (scalar_t(i) - center) / (std * center);
  return std::exp(scalar_t(-0.5) * x * x);
}

// Gradient w.r.t. std
template<typename scalar_t>
inline scalar_t gaussian_backward(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  scalar_t std,
  bool periodic,
  scalar_t forward_value
) {
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  scalar_t center = denom / scalar_t(2);
  scalar_t x = (scalar_t(i) - center) / (std * center);
  // d/d(std) of exp(-0.5 * x^2) where x = (i - center) / (std * center)
  // = forward_value * x^2 / std
  return grad_out * forward_value * x * x / std;
}

}  // namespace torchscience::kernel::window_function
```

### Adding a New Window

**Parameterless window (e.g., Blackman):**

1. Add kernel (~8 lines):
```cpp
// kernel/signal_processing/window_function/blackman.h
template<typename scalar_t>
inline scalar_t blackman(int64_t i, int64_t n, bool periodic) {
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == 0) return scalar_t(1);
  scalar_t x = scalar_t(2) * M_PI * scalar_t(i) / denom;
  return scalar_t(0.42) - scalar_t(0.5)*std::cos(x) + scalar_t(0.08)*std::cos(2*x);
}
```

2. Add macro invocations (3 lines across files):
```cpp
// cpu/window_functions.h
TORCHSCIENCE_CPU_WINDOW_PARAMETERLESS(blackman)

// meta/window_functions.h
TORCHSCIENCE_META_WINDOW_PARAMETERLESS(blackman)
```

**Total: ~11 lines for a new parameterless window**

**Parameterized window (e.g., Kaiser):**

1. Add kernel with backward (~20 lines):
```cpp
// kernel/signal_processing/window_function/kaiser.h
template<typename scalar_t>
inline scalar_t kaiser(int64_t i, int64_t n, scalar_t beta, bool periodic) {
  // Implementation using modified Bessel function I0
  // ...
}

template<typename scalar_t>
inline scalar_t kaiser_backward(
  scalar_t grad_out, int64_t i, int64_t n,
  scalar_t beta, bool periodic, scalar_t fwd_value
) {
  // Gradient w.r.t. beta
  // ...
}
```

2. Add macro invocations (4 lines across files):
```cpp
// cpu/window_functions.h
TORCHSCIENCE_CPU_WINDOW_1PARAM(kaiser, beta)

// meta/window_functions.h
TORCHSCIENCE_META_WINDOW_1PARAM(kaiser, beta)

// autograd/window_functions.h
TORCHSCIENCE_AUTOGRAD_WINDOW_1PARAM(kaiser, Kaiser, beta)
```

**Total: ~24 lines for a new parameterized window**

### Comparison: Macro vs Non-Macro

| Approach | Lines per parameterless | Lines per parameterized |
|----------|------------------------|-------------------------|
| Current design doc (non-macro) | ~12 | ~50+ |
| **Macro-based** | **~11** | **~24** |
| Existing pointwise operators | N/A (different pattern) | N/A |

The macro approach provides:
- **Consistent structure** across CPU/Meta/Autograd backends
- **Reduced duplication** — change once in macro, applies everywhere
- **Easier debugging** — all implementations follow same pattern
- **Faster iteration** — adding a window is mechanical, not creative

## Python Module Structure

```
src/torchscience/signal_processing/window_function/
├── __init__.py
├── _bartlett_window.py
├── _periodic_bartlett_window.py
├── _blackman_window.py
├── _periodic_blackman_window.py
├── _cosine_window.py
├── _periodic_cosine_window.py
├── _gaussian_window.py
├── _periodic_gaussian_window.py
├── _hamming_window.py
├── _periodic_hamming_window.py
├── _hann_window.py
├── _periodic_hann_window.py
├── _nuttall_window.py
├── _periodic_nuttall_window.py
├── _general_cosine_window.py
├── _periodic_general_cosine_window.py
├── _general_hamming_window.py
├── _periodic_general_hamming_window.py
└── _rectangular_window.py          # existing
```

## Operator Registration

### Schema Definitions

Schemas are defined in `torchscience.cpp`. The macros handle all `TORCH_LIBRARY_IMPL` registration automatically.

```cpp
// In torchscience.cpp TORCH_LIBRARY block:

// Parameterless windows
m.def("hann_window(int n, ScalarType? dtype=None, Layout? layout=None, "
      "Device? device=None, bool requires_grad=False) -> Tensor");
m.def("periodic_hann_window(int n, ScalarType? dtype=None, Layout? layout=None, "
      "Device? device=None, bool requires_grad=False) -> Tensor");
// ... repeat for hamming, blackman, bartlett, cosine, nuttall

// Parameterized windows (1 parameter)
m.def("gaussian_window(int n, Tensor std, ScalarType? dtype=None, "
      "Layout? layout=None, Device? device=None) -> Tensor");
m.def("periodic_gaussian_window(int n, Tensor std, ScalarType? dtype=None, "
      "Layout? layout=None, Device? device=None) -> Tensor");
m.def("gaussian_window_backward(Tensor grad_output, Tensor output, int n, "
      "Tensor std, bool periodic) -> Tensor");
m.def("periodic_gaussian_window_backward(Tensor grad_output, Tensor output, int n, "
      "Tensor std, bool periodic) -> Tensor");
// ... repeat for kaiser, general_hamming

// Multi-parameter windows
m.def("general_cosine_window(int n, Tensor coeffs, ScalarType? dtype=None, "
      "Layout? layout=None, Device? device=None) -> Tensor");
m.def("periodic_general_cosine_window(int n, Tensor coeffs, ScalarType? dtype=None, "
      "Layout? layout=None, Device? device=None) -> Tensor");
m.def("general_cosine_window_backward(Tensor grad_output, Tensor output, int n, "
      "Tensor coeffs, bool periodic) -> Tensor");
m.def("periodic_general_cosine_window_backward(Tensor grad_output, Tensor output, int n, "
      "Tensor coeffs, bool periodic) -> Tensor");
```

### Key Design Notes

**Why `int n` instead of `Scalar n`:**
- Simpler schema, cleaner API
- Variable-length batching (NestedTensor output) can be added as a separate API if needed
- Matches `torch.hann_window(window_length)` convention

**Parameterless windows:**
- No backward pass needed — `requires_grad` only affects output tensor
- CPU macro registers both symmetric and periodic variants from single kernel

**Parameterized windows:**
- Autograd wrapper intercepts forward, calls CPU kernel via dispatcher
- Backward registered separately for each variant (symmetric/periodic)

### Autograd Details

The `TORCHSCIENCE_AUTOGRAD_WINDOW_1PARAM` macro (see C++ Architecture section) generates:

1. **Forward class** (`Name##Window`) that:
   - Dispatches to CPU kernel via c10::Dispatcher
   - Saves output and parameter tensors for backward
   - Stores `n`, `periodic`, and `param_requires_grad` in `saved_data`

2. **Backward class** (`Name##WindowBackward`) that:
   - Dispatches to `name_window_backward` kernel
   - Returns gradient w.r.t. parameter tensor
   - Second-order gradients return empty tensors (typically not needed for window functions)

3. **Inline wrapper functions** that call the autograd class's `apply()` method

## Testing Strategy

### Test Infrastructure

```python
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import torch
from torch import Tensor
from torch.testing import assert_close
import pytest


@dataclass
class WindowOpDescriptor:
    """Describes a window function for parametrized testing."""
    name: str
    func: Callable
    reference_func: Callable  # Python reference implementation
    pytorch_func: Optional[Callable] = None  # torch.signal.windows.* if available
    parameters: List[str] = None  # e.g., ["std"] for gaussian
    is_periodic: bool = False
    supported_dtypes: List[torch.dtype] = None

    def __post_init__(self):
        self.parameters = self.parameters or []
        self.supported_dtypes = self.supported_dtypes or [
            torch.float32, torch.float64, torch.bfloat16, torch.float16
        ]


class TestWindowBase:
    """Base class for window function tests with shared fixtures."""

    @pytest.fixture
    def window_sizes(self):
        return [0, 1, 2, 5, 64, 1024]

    @pytest.fixture
    def batch_sizes(self):
        return [1, 4, 16]

    # -------------------------------------------------------------------------
    # Basic correctness tests
    # -------------------------------------------------------------------------

    def test_reference_comparison(self, descriptor, window_sizes):
        """Compare against Python reference implementation."""
        for n in window_sizes:
            result = descriptor.func(n)
            expected = descriptor.reference_func(n)
            assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_pytorch_comparison(self, descriptor, window_sizes):
        """Compare against torch.signal.windows.* where available."""
        if descriptor.pytorch_func is None:
            pytest.skip("No PyTorch equivalent")
        for n in window_sizes:
            if n == 0:
                continue  # PyTorch may not support n=0
            result = descriptor.func(n)
            expected = descriptor.pytorch_func(n, sym=not descriptor.is_periodic)
            assert_close(result, expected)

    # -------------------------------------------------------------------------
    # Shape and dtype tests
    # -------------------------------------------------------------------------

    def test_output_shape_scalar_n(self, descriptor, window_sizes):
        """Scalar n produces (n,) output."""
        for n in window_sizes:
            result = descriptor.func(n)
            assert result.shape == (n,)

    def test_output_shape_batched_n(self, descriptor):
        """Tensor n produces NestedTensor output."""
        n_tensor = torch.tensor([5, 8, 3])
        result = descriptor.func(n_tensor)
        assert result.is_nested
        unbind = result.unbind()
        assert len(unbind) == 3
        assert unbind[0].shape == (5,)
        assert unbind[1].shape == (8,)
        assert unbind[2].shape == (3,)

    def test_dtype_support(self, descriptor, window_sizes):
        """Test all supported dtypes."""
        for dtype in descriptor.supported_dtypes:
            result = descriptor.func(64, dtype=dtype)
            assert result.dtype == dtype

    def test_device_support(self, descriptor):
        """Test CPU and CUDA (if available)."""
        result_cpu = descriptor.func(64, device="cpu")
        assert result_cpu.device.type == "cpu"

        if torch.cuda.is_available():
            result_cuda = descriptor.func(64, device="cuda")
            assert result_cuda.device.type == "cuda"
            assert_close(result_cpu, result_cuda.cpu())

    # -------------------------------------------------------------------------
    # Autograd tests (for parameterized windows)
    # -------------------------------------------------------------------------

    def test_gradcheck(self, descriptor):
        """Test gradient correctness with torch.autograd.gradcheck."""
        if not descriptor.parameters:
            pytest.skip("Parameterless window")

        # Create parameter tensor with requires_grad
        param = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda p: descriptor.func(64, p).sum(),
            (param,),
            raise_exception=True
        )

    def test_gradgradcheck(self, descriptor):
        """Test second-order gradient correctness."""
        if not descriptor.parameters:
            pytest.skip("Parameterless window")

        param = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            lambda p: descriptor.func(64, p).sum(),
            (param,),
            raise_exception=True
        )

    # -------------------------------------------------------------------------
    # NestedTensor-specific tests
    # -------------------------------------------------------------------------

    def test_nested_tensor_unbind(self, descriptor):
        """NestedTensor can be unbound into individual windows."""
        n_tensor = torch.tensor([10, 20, 15])
        result = descriptor.func(n_tensor)

        windows = result.unbind()
        assert len(windows) == 3

        # Each window should match single-window result
        for i, (n, window) in enumerate(zip([10, 20, 15], windows)):
            expected = descriptor.func(n)
            assert_close(window, expected)

    def test_nested_tensor_to_padded(self, descriptor):
        """NestedTensor can be converted to padded dense tensor."""
        n_tensor = torch.tensor([5, 8, 3])
        result = descriptor.func(n_tensor)

        # Convert to padded tensor
        padded = result.to_padded_tensor(padding=0.0)
        assert padded.shape == (3, 8)  # (batch, max_len)

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_n_equals_zero(self, descriptor):
        """n=0 produces empty tensor."""
        result = descriptor.func(0)
        assert result.shape == (0,)
        assert result.numel() == 0

    def test_n_equals_one(self, descriptor):
        """n=1 produces single-element window."""
        result = descriptor.func(1)
        assert result.shape == (1,)
        # Most windows have w[0] = 1 for n=1
        # (or some other well-defined value)

    def test_symmetry_property(self, descriptor):
        """Symmetric windows are mirror-symmetric around center."""
        if descriptor.is_periodic:
            pytest.skip("Periodic windows are not symmetric")

        n = 65  # Odd length for exact center
        result = descriptor.func(n)

        # Check w[i] == w[n-1-i]
        for i in range(n // 2):
            assert_close(result[i], result[n - 1 - i], atol=1e-10, rtol=0)
```

### Test Coverage Matrix

| Test Category | Parameterless | Parameterized | NestedTensor |
|---------------|---------------|---------------|--------------|
| Reference comparison | ✓ | ✓ | ✓ |
| PyTorch comparison | ✓ | ✓ | N/A |
| Shape (scalar n) | ✓ | ✓ | N/A |
| Shape (Tensor n) | ✓ | ✓ | ✓ |
| Dtype support | ✓ | ✓ | ✓ |
| Device support | ✓ | ✓ | ✓ |
| Meta tensor | ✓ | ✓ | ✓ |
| gradcheck | N/A | ✓ | ✓ |
| gradgradcheck | N/A | ✓ | ✓ |
| torch.compile | ✓ | ✓ | TBD |
| n=0, n=1 | ✓ | ✓ | ✓ |
| Symmetry | ✓ | ✓ | ✓ |
| unbind/to_padded | N/A | N/A | ✓ |

## Mathematical Definitions

### Hann Window (Symmetric)

```
w[k] = 0.5 * (1 - cos(2πk / (n-1))),  for k = 0, 1, ..., n-1
```

### Hann Window (Periodic)

```
w[k] = 0.5 * (1 - cos(2πk / n)),  for k = 0, 1, ..., n-1
```

### Gaussian Window

```
w[k] = exp(-0.5 * ((k - (n-1)/2) / (std * (n-1)/2))^2)
```

### General Cosine Window

```
w[k] = sum_{i=0}^{M-1} (-1)^i * a_i * cos(2πik / (n-1))
```

### General Hamming Window

```
w[k] = alpha - (1 - alpha) * cos(2πk / (n-1))
```

## Implementation Order

### Phase 1 Priority

1. `hann_window` / `periodic_hann_window` — most common
2. `hamming_window` / `periodic_hamming_window` — second most common
3. `blackman_window` / `periodic_blackman_window` — 3-term cosine
4. `bartlett_window` / `periodic_bartlett_window` — triangular
5. `cosine_window` / `periodic_cosine_window` — half-cosine
6. `nuttall_window` / `periodic_nuttall_window` — 4-term cosine
7. `gaussian_window` / `periodic_gaussian_window` — first parameterized
8. `general_hamming_window` / `periodic_general_hamming_window` — parameterized
9. `general_cosine_window` / `periodic_general_cosine_window` — N-term parameterized

## Risks and Considerations

### NestedTensor Maturity

NestedTensor is still evolving in PyTorch. Considerations:

1. **API stability** — NestedTensor API may change between PyTorch versions
2. **CUDA support** — Some NestedTensor operations may not be fully optimized on GPU
3. **torch.compile** — NestedTensor support in torch.compile is experimental

**Mitigation:** Document PyTorch version requirements, provide `to_padded_tensor()` escape hatch.

### Macro Maintenance

Using macros for all window registrations:

1. **Debugging difficulty** — Macro-expanded code is harder to step through in debugger
2. **Error messages** — Compilation errors point to macro definition, not invocation site

**Mitigation:** Keep macros focused and well-documented. Use `#line` directives if needed.

### Performance with NestedTensor

Creating NestedTensor from list of tensors has overhead:

1. **Memory allocation** — Each window is a separate allocation
2. **Kernel launch** — Multiple small kernels vs one large kernel

**Mitigation:** For same-length batches, detect and use optimized `(B, n)` path.

## References

- Harris, F.J. "On the use of windows for harmonic analysis with the discrete Fourier transform," Proceedings of the IEEE, vol. 66, no. 1, pp. 51-83, Jan. 1978.
- Oppenheim, A.V. and Schafer, R.W. "Discrete-Time Signal Processing," 3rd ed., Prentice Hall, 2009.
- [PyTorch torch.signal.windows](https://docs.pytorch.org/docs/stable/signal.html)
- [SciPy scipy.signal.windows](https://docs.scipy.org/doc/scipy/reference/signal.windows.html)
- [PyTorch NestedTensor](https://pytorch.org/docs/stable/nested.html)
