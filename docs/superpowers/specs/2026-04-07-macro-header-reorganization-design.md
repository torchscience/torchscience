# Macro Header Reorganization

## Problem

The current macro system has four pain points:

1. **File size** — `cpu/macros.h` is 335KB. Working in it is unwieldy.
2. **Discoverability** — Choosing between `_WITH_COMPLEX` vs plain, `_EX` vs base, and five arities is confusing. The naming doesn't guide you to the right macro.
3. **Extensibility** — Macros are coupled to the `special_functions` kernel namespace. Adding a new category (graphics, statistics) requires duplicating macros or hacking around the hardcoded namespace.
4. **Duplication** — The forward/backward/backward_backward pattern repeats across every backend's macros with minor variations.

## Design

### File Structure

Macros move from per-backend files (`cpu/macros.h`, `autograd/macros.h`) to a dedicated `csrc/macros/` directory, organized by backend then operator type:

```
csrc/macros/
├── cpu/
│   ├── pointwise.h
│   ├── reduction.h
│   ├── creation.h
│   └── identity.h
├── cuda/
│   ├── pointwise.cuh
│   ├── reduction.cuh
│   ├── creation.cuh
│   └── identity.cuh
├── meta/
│   ├── pointwise.h
│   ├── reduction.h
│   ├── creation.h
│   └── identity.h
├── autograd/
│   ├── pointwise.h
│   ├── reduction.h
│   ├── reduction_helpers.h
│   └── identity.h
├── autocast/
│   ├── pointwise.h
│   ├── reduction.h
│   └── identity.h
├── batched/
│   ├── pointwise.h
│   ├── reduction.h
│   └── identity.h
├── sparse/
│   ├── coo/
│   │   ├── cpu/
│   │   │   ├── pointwise.h
│   │   │   ├── reduction.h
│   │   │   └── identity.h
│   │   └── cuda/
│   │       ├── pointwise.cuh
│   │       ├── reduction.cuh
│   │       └── identity.cuh
│   └── csr/
│       ├── cpu/
│       │   ├── pointwise.h
│       │   ├── reduction.h
│       │   └── identity.h
│       └── cuda/
│           ├── pointwise.cuh
│           ├── reduction.cuh
│           └── identity.cuh
├── nested/
│   ├── cpu/
│   │   ├── pointwise.h
│   │   ├── reduction.h
│   │   └── identity.h
│   └── cuda/
│       ├── pointwise.cuh
│       ├── reduction.cuh
│       └── identity.cuh
└── quantized/
    ├── cpu/
    │   ├── pointwise.h
    │   ├── reduction.h
    │   └── identity.h
    └── cuda/
        ├── pointwise.cuh
        ├── reduction.cuh
        └── identity.cuh
```

The old files (`cpu/macros.h`, `cpu/reduction_macros.h`, `autograd/macros.h`, etc.) are deleted. Autograd's `TSCI_*` helper macros (`TSCI_EXTRA`, `TSCI_SAVE`, `TSCI_LOAD`, etc.) move to `macros/autograd/reduction_helpers.h`.

### Macro Signatures

#### Naming Convention

General pattern for preprocessor macros:

```
TORCHSCIENCE_{BACKEND}_{OPERATOR_TYPE}_{ARITY}(category, complex, name, args...)
```

Each operator type adapts this pattern to its needs:

- **Pointwise**: follows the pattern exactly — `(category, complex, name, args...)`
- **Reduction**: drops `complex`, adds `dim`/`keepdim` or `mode` — `(category, name, x, dim, keepdim)`
- **Identity**: adds `dim` — `(category, complex, name, x, dim)`
- **Creation**: uses templates instead of preprocessor macros — `(name, Traits, ...)`

Changes from the current convention:

- `_OPERATOR` suffix dropped (the file name communicates that these are macros)
- `_WITH_COMPLEX` suffix eliminated (replaced by boolean `complex` parameter)
- `category` parameter added (replaces hardcoded `kernel::special_functions::` namespace)

#### Pointwise Macros

```cpp
TORCHSCIENCE_{BACKEND}_POINTWISE_UNARY(category, complex, name, a)
TORCHSCIENCE_{BACKEND}_POINTWISE_BINARY(category, complex, name, a, b)
TORCHSCIENCE_{BACKEND}_POINTWISE_TERNARY(category, complex, name, a, b, c)
TORCHSCIENCE_{BACKEND}_POINTWISE_QUATERNARY(category, complex, name, a, b, c, d)
TORCHSCIENCE_{BACKEND}_POINTWISE_QUINARY(category, complex, name, a, b, c, d, e)
```

Where `{BACKEND}` is one of: `CPU`, `CUDA`, `META`, `AUTOGRAD`, `AUTOCAST`, `BATCHED`, `SPARSE_COO_CPU`, `SPARSE_COO_CUDA`, `SPARSE_CSR_CPU`, `SPARSE_CSR_CUDA`, `NESTED_CPU`, `NESTED_CUDA`, `QUANTIZED_CPU`, `QUANTIZED_CUDA`.

#### Reduction Macros

```cpp
// Dim-based: reduces over user-specified dimensions
TORCHSCIENCE_{BACKEND}_DIM_REDUCTION_UNARY(category, name, x, dim, keepdim)
TORCHSCIENCE_{BACKEND}_DIM_REDUCTION_UNARY_EX(category, name, x, dim, keepdim, ...)

// Fixed: reduces over predetermined dimensions
TORCHSCIENCE_{BACKEND}_FIXED_REDUCTION_UNARY(category, name, mode, x)
TORCHSCIENCE_{BACKEND}_FIXED_REDUCTION_UNARY_EX(category, name, mode, x, ...)
```

Changes from the current reduction macros:

- `NS` parameter renamed to `category` (consistency with pointwise)
- `_OPERATOR` suffix dropped
- `MODE` parameter stays for FIXED variants (`ReductionMode::LAST_DIM` or `ReductionMode::ALL_DIMS`)

`_EX` variants are retained — they handle genuinely different parameter shapes (extra bool/int/double arguments) that a boolean flag cannot collapse.

Reduction macros do not take a `complex` parameter. Reductions collapse tensor dimensions — the dispatch type is determined by the input tensor's dtype at the CPU/CUDA kernel level, not by a macro-level flag. The current reduction macros already handle all floating types uniformly.

**TSCI_\* helpers** (`TSCI_EXTRA`, `TSCI_SAVE`, `TSCI_LOAD`, `TSCI_EXTRA_2BOOL`, etc.) move to `macros/autograd/reduction_helpers.h`. They are only consumed by autograd reduction macros. The helpers themselves are unchanged.

**Backend-specific behavior:**

| Backend | What it generates |
|---------|------------------|
| **CPU** | Forward with `at::parallel_for` over batch dims, `AT_DISPATCH_FLOATING_TYPES_AND2` type dispatch, calls `kernel::category::name<scalar_t>()`. Backward and backward_backward follow the same parallel pattern. |
| **CUDA** | Same structure as CPU but with `CUDAGuard`, `gpu_kernel`, `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2`. |
| **Meta** | Shape inference only — computes output shape via `compute_reduction_shape()`, returns empty tensors. Extra params marked `[[maybe_unused]]`. |
| **Autograd** | Generates two `torch::autograd::Function` subclasses (`Name` + `Name##Backward`) with context save/load for extra params. Dispatches to CPU/CUDA backward via `c10::Dispatcher`. |
| **Autocast** | Casts input to float32 with `ExcludeDispatchKeyGuard`, redispatches. |
| **Batched** | vmap support — reshapes batch dims, delegates to underlying dispatch. |

**Autograd _EX call site** (the most complex case):

```cpp
TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_EX(
    statistics::descriptive, kurtosis, Kurtosis, input,
    TSCI_EXTRA_2BOOL(fisher, bias)
)
```

`TSCI_EXTRA_2BOOL` expands to all six extra-parameter macro arguments (EXTRA_PARAMS, EXTRA_ARGS, EXTRA_TYPES, EXTRA_SAVE, EXTRA_LOAD, EXTRA_GRAD_PLACEHOLDERS).

#### Creation Macros

Creation operators use a template-based approach rather than preprocessor macros. The templates and their registration macros live in `macros/` but the heavy lifting is in C++ templates parameterized by a `Traits` struct.

**Template classes:**

| Class | Location | Role |
|-------|----------|------|
| `CPUCreationOperator<Traits>` | `macros/cpu/creation.h` | Deterministic CPU creation |
| `CPUStochasticCreationOperator<Traits>` | `macros/cpu/creation.h` | RNG-based CPU creation |
| `CUDACreationOperator<Traits>` | `macros/cuda/creation.cuh` | Deterministic CUDA creation |
| `CUDAStochasticCreationOperator<Traits>` | `macros/cuda/creation.cuh` | RNG-based CUDA creation |
| `MetaCreationOperator<Traits>` | `macros/meta/creation.h` | Shape inference only |

The `Traits` contract:

```cpp
struct MyTraits {
    static std::vector<int64_t> output_shape(params...);
    template<typename scalar_t>
    static void kernel(scalar_t* out, int64_t numel, params...);
};
```

Stochastic traits add an RNG parameter to `kernel`.

**Registration macros:**

```cpp
// macros/cpu/creation.h
#define TORCHSCIENCE_CPU_CREATION(name, Traits, ...) \
    TORCH_LIBRARY_IMPL(torchscience, CPU, _m_cpu_##name) { \
        _m_cpu_##name.impl(#name, \
            &::torchscience::cpu::CPUCreationOperator<Traits>::forward<__VA_ARGS__>); \
    }

#define TORCHSCIENCE_CPU_STOCHASTIC_CREATION(name, Traits, ...) \
    TORCH_LIBRARY_IMPL(torchscience, CPU, _m_cpu_##name) { \
        _m_cpu_##name.impl(#name, \
            &::torchscience::cpu::CPUStochasticCreationOperator<Traits>::forward<__VA_ARGS__>); \
    }
```

Same pattern for CUDA and Meta. Each invocation self-registers (generates its own `TORCH_LIBRARY_IMPL` block).

**No `category` parameter:** Unlike pointwise/reduction, the template approach doesn't generate `kernel::category::name()` calls — the `Traits` struct already encapsulates which kernel to call.

**Backends that don't apply:** Creation operators create tensors from scalar parameters — no input tensors. Autograd, Autocast, Batched, Sparse, Nested, and Quantized backends do not apply. Only CPU, CUDA, and Meta backends exist. The corresponding files from the directory structure are removed.

#### Identity Macros

Identity operators are unary, shape-preserving, and operate on slices of a specified dimension. They use pointer-based kernels rather than scalar lambdas.

```cpp
TORCHSCIENCE_{BACKEND}_IDENTITY_UNARY(category, complex, name, x, dim)
```

- `category`: kernel namespace (e.g., `graphics`)
- `complex`: type dispatch selector (token-pasted, same mechanism as pointwise)
- `name`: operator name (e.g., `srgb_to_hsv`)
- `x`: input tensor argument name
- `dim`: operating dimension argument name (`int64_t` in the generated function)

**Kernel interface:**

Kernels receive pointers to one contiguous slice of the operating dimension. The macro handles permutation — kernels always see a contiguous slice:

```cpp
namespace torchscience::kernel::graphics {

// Forward: reads in[0..D-1], writes out[0..D-1]
template<typename T>
void srgb_to_hsv(const T* in, T* out, int64_t dim_size);

// Backward: reads grad_out and in, writes grad_in
template<typename T>
void srgb_to_hsv_backward(
    const T* grad_out, const T* in, T* grad_in, int64_t dim_size);

// Backward²: reads grad_grad_in, grad_out, in; writes grad_grad_out, new_grad_in
template<typename T>
void srgb_to_hsv_backward_backward(
    const T* grad_grad_in, const T* grad_out, const T* in,
    T* grad_grad_out, T* new_grad_in, int64_t dim_size);
}
```

**CPU generated code (forward):**

1. Normalize `dim` via `at::maybe_wrap_dim`
2. Make input contiguous
3. If `dim != last`: permute input to move `dim` to the last position
4. Compute `batch_size` (product of all dims except last) and `dim_size`
5. Allocate output with permuted shape
6. `AT_DISPATCH_FLOATING_TYPES_AND2` (or `_AND_COMPLEX_` if `complex=true`)
7. `at::parallel_for(0, batch_size, ...)` — for each batch index `i`, call `kernel::category::name<scalar_t>(in + i*D, out + i*D, D)`
8. If `dim != last`: permute output back to original dimension ordering

Backward and backward_backward apply the same permute-in / kernel / permute-out pattern.

**Backend-specific behavior:**

| Backend | Behavior |
|---------|----------|
| **CPU** | Contiguous + parallel_for over batch dims, scalar type dispatch, pointer-based kernel calls |
| **CUDA** | CUDAGuard + GPU kernel launch, each thread block processes one or more dim slices |
| **Meta** | Returns `at::empty_like(x)` — no permutation needed since shape is preserved |
| **Autograd** | Two Function classes (`Name` + `Name##Backward`), saves `x` and `dim` in context, dispatches to backward/backward_backward via `c10::Dispatcher` |
| **Autocast** | Casts input to lower precision, excludes autocast key, redispatches, passes `dim` through |
| **Batched** | vmap support — batch dims are additional leading dimensions, same iteration applies |

### Parameter Design

#### `category` Parameter

Controls kernel namespace resolution. The macro generates calls to `kernel::category::name(args...)`:

```cpp
// Expands to kernel::special_functions::gamma(z)
TORCHSCIENCE_CPU_POINTWISE_UNARY(special_functions, true, gamma, z)

// Expands to kernel::graphics::srgb_to_linear(x)
TORCHSCIENCE_CPU_POINTWISE_UNARY(graphics, false, srgb_to_linear, x)
```

#### `complex` Parameter

A boolean literal (`true`/`false`) that selects the type dispatch variant at compile time via token pasting:

```cpp
#define TORCHSCIENCE_CPU_POINTWISE_UNARY(category, complex, name, a) \
    TORCHSCIENCE_CPU_POINTWISE_UNARY_DISPATCH_##complex(category, name, a)

#define TORCHSCIENCE_CPU_POINTWISE_UNARY_DISPATCH_true(category, name, a) \
    /* ... */ \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2( \
        at::kBFloat16, at::kHalf, /* ... */) { \
        cpu_kernel(/* ... */, [](scalar_t a) -> scalar_t { \
            return kernel::category::name(a); \
        }); \
    } \
    /* ... */

#define TORCHSCIENCE_CPU_POINTWISE_UNARY_DISPATCH_false(category, name, a) \
    /* ... */ \
    AT_DISPATCH_FLOATING_TYPES_AND2( \
        at::kBFloat16, at::kHalf, /* ... */) { \
        cpu_kernel(/* ... */, [](scalar_t a) -> scalar_t { \
            return kernel::category::name(a); \
        }); \
    } \
    /* ... */
```

No runtime cost. No conditional compilation complexity. The boolean becomes part of the macro name at preprocessing time.

**Backend-specific behavior:**

- **CPU**: `complex` selects between `AT_DISPATCH_FLOATING_TYPES_AND2` and `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2`
- **CUDA**: `complex` is accepted for signature consistency but ignored — CUDA always dispatches over complex types (matching current behavior)
- **Meta, Autograd, Autocast, Batched**: `complex` is accepted but ignored — these backends are type-agnostic (Meta does shape inference, Autograd/Autocast/Batched delegate to underlying dispatch)
- **Sparse COO/CSR, Nested, Quantized**: `complex` selects dispatch variants, same as CPU — these backends run actual compute on their respective tensor layouts

### Registration Files (Call Sites)

Registration files (`cpu/special_functions.h`, `autograd/special_functions.h`, etc.) stay in their current locations. They update their includes and macro invocations:

```cpp
// cpu/special_functions.h — before
#include "macros.h"
#include "../kernel/special_functions/gamma.h"
#include "../kernel/special_functions/gamma_backward.h"
#include "../kernel/special_functions/gamma_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(gamma, z)
TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_t, n, x)
```

```cpp
// cpu/special_functions.h — after
#include "../macros/cpu/pointwise.h"
#include "../kernel/special_functions/gamma.h"
#include "../kernel/special_functions/gamma_backward.h"
#include "../kernel/special_functions/gamma_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY(special_functions, true,  gamma, z)
TORCHSCIENCE_CPU_POINTWISE_BINARY(special_functions, false, chebyshev_polynomial_t, n, x)
```

New categories add a new registration file per backend:

```cpp
// cpu/graphics.h
#include "../macros/cpu/pointwise.h"
#include "../macros/cpu/identity.h"
#include "../kernel/graphics/srgb_to_linear.h"
#include "../kernel/graphics/srgb_to_hsv.h"
// ...

TORCHSCIENCE_CPU_POINTWISE_UNARY(graphics, false, srgb_to_linear, x)
TORCHSCIENCE_CPU_IDENTITY_UNARY(graphics, false, srgb_to_hsv, input, dim)
```

Reduction operators follow the same pattern:

```cpp
// cpu/statistics.h
#include "../macros/cpu/reduction.h"
#include "../kernel/statistics/descriptive/kurtosis.h"
// ...

TORCHSCIENCE_CPU_DIM_REDUCTION_UNARY_EX(
    statistics::descriptive, kurtosis, input,
    TSCI_EXTRA(bool fisher, bool bias),
    TSCI_EXTRA(fisher, bias)
)
```

Creation operators use templates with thin registration macros:

```cpp
// cpu/signal_processing.h
#include "../macros/cpu/creation.h"
#include "../kernel/signal_processing/rectangular_window.h"
// ...

TORCHSCIENCE_CPU_CREATION(rectangular_window, RectangularWindowTraits, int64_t)
TORCHSCIENCE_CPU_STOCHASTIC_CREATION(pink_noise, PinkNoiseTraits, int64_t)
```

The same macro headers are reused across categories. No new macro definitions needed.

### Backends

| Backend | Directory | Dispatch Key | Role |
|---------|-----------|-------------|------|
| CPU | `macros/cpu/` | `CPU` | Compute kernels |
| CUDA | `macros/cuda/` | `CUDA` | GPU compute kernels |
| Meta | `macros/meta/` | `Meta` | Shape inference (no computation) |
| Autograd | `macros/autograd/` | `AutogradCPU`, `AutogradCUDA` | Gradient wrappers via `torch::autograd::Function` |
| Autocast | `macros/autocast/` | `AutocastCPU`, `AutocastCUDA` | Mixed precision casting |
| Batched | `macros/batched/` | `FuncTorchBatched` | vmap support |
| Sparse COO | `macros/sparse/coo/{cpu,cuda}/` | `SparseCPU`, `SparseCUDA` | COO sparse tensor dispatch |
| Sparse CSR | `macros/sparse/csr/{cpu,cuda}/` | `SparseCsrCPU`, `SparseCsrCUDA` | CSR sparse tensor dispatch |
| Nested | `macros/nested/{cpu,cuda}/` | `NestedTensorCPU`, `NestedTensorCUDA` | Nested/ragged tensor dispatch |
| Quantized | `macros/quantized/{cpu,cuda}/` | `QuantizedCPU`, `QuantizedCUDA` | Dequantize, compute, requantize |

### Operator Types

| Type | File | Shape Behavior | Applicable Backends | Examples |
|------|------|---------------|---------------------|----------|
| Pointwise | `pointwise.h` | Element-wise with broadcasting | All | `gamma`, `beta`, `binomial_coefficient` |
| Reduction | `reduction.h` | Reduces dimensions | All | `kurtosis`, `kullback_leibler_divergence` |
| Creation | `creation.h` | Creates tensors from parameters | CPU, CUDA, Meta only | `rectangular_window`, `pink_noise` |
| Identity | `identity.h` | Preserves shape exactly | All | `srgb_to_hsv` |

Fixed, batched, and dynamic operator types are deferred until enough operators exist to establish a stable pattern.

### What Does Not Change

- **Kernel headers** (`csrc/kernel/`) — untouched, these are the actual implementations
- **Schema definitions** (`csrc/torchscience.cpp`) — untouched
- **Python wrappers** (`src/torchscience/<module>/`) — untouched
- **Registration file locations** (`cpu/special_functions.h`, etc.) — same paths, updated contents
- **Generated function signatures** — the macros produce the same forward/backward/backward_backward functions with the same dispatch registrations
