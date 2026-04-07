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
│   ├── creation.h
│   └── identity.h
├── autocast/
│   ├── pointwise.h
│   ├── reduction.h
│   ├── creation.h
│   └── identity.h
├── batched/
│   ├── pointwise.h
│   ├── reduction.h
│   ├── creation.h
│   └── identity.h
├── sparse/
│   ├── coo/
│   │   ├── cpu/
│   │   │   ├── pointwise.h
│   │   │   ├── reduction.h
│   │   │   ├── creation.h
│   │   │   └── identity.h
│   │   └── cuda/
│   │       ├── pointwise.cuh
│   │       ├── reduction.cuh
│   │       ├── creation.cuh
│   │       └── identity.cuh
│   └── csr/
│       ├── cpu/
│       │   ├── pointwise.h
│       │   ├── reduction.h
│       │   ├── creation.h
│       │   └── identity.h
│       └── cuda/
│           ├── pointwise.cuh
│           ├── reduction.cuh
│           ├── creation.cuh
│           └── identity.cuh
├── nested/
│   ├── cpu/
│   │   ├── pointwise.h
│   │   ├── reduction.h
│   │   ├── creation.h
│   │   └── identity.h
│   └── cuda/
│       ├── pointwise.cuh
│       ├── reduction.cuh
│       ├── creation.cuh
│       └── identity.cuh
└── quantized/
    ├── cpu/
    │   ├── pointwise.h
    │   ├── reduction.h
    │   ├── creation.h
    │   └── identity.h
    └── cuda/
        ├── pointwise.cuh
        ├── reduction.cuh
        ├── creation.cuh
        └── identity.cuh
```

The old files (`cpu/macros.h`, `cpu/reduction_macros.h`, `autograd/macros.h`, etc.) are deleted. Autograd's `TSCI_*` helper macros (`TSCI_EXTRA`, `TSCI_SAVE`, `TSCI_LOAD`, etc.) move to `macros/autograd/reduction_helpers.h`.

### Macro Signatures

#### Naming Convention

```
TORCHSCIENCE_{BACKEND}_{OPERATOR_TYPE}_{ARITY}(category, complex, name, args...)
```

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
TORCHSCIENCE_{BACKEND}_DIM_REDUCTION_UNARY(category, name, x, dim, keepdim)
TORCHSCIENCE_{BACKEND}_DIM_REDUCTION_UNARY_EX(category, name, x, dim, keepdim, ...)
TORCHSCIENCE_{BACKEND}_FIXED_REDUCTION_UNARY(category, name, x, ...)
TORCHSCIENCE_{BACKEND}_FIXED_REDUCTION_UNARY_EX(category, name, x, ...)
```

`_EX` variants are retained — they handle genuinely different parameter shapes (extra bool/int arguments) that a boolean flag cannot collapse.

Reduction macros do not take a `complex` parameter. Reductions collapse tensor dimensions — the dispatch type is determined by the input tensor's dtype at the CPU/CUDA kernel level, not by a macro-level flag. The current reduction macros already handle all floating types uniformly.

#### Creation and Identity Macros

Exact parameter lists are deferred to implementation — no operators currently use these macro types, so the signatures will be designed when the first operators are added. The design constraints are established:

- Creation macros will follow `(category, name, ...)` with additional dtype, device, and layout parameters as needed by factory operators.
- Identity macros will follow `(category, complex, name, args...)` matching the pointwise signature pattern but without broadcasting in the generated `TensorIteratorConfig`.

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
#include "../kernel/graphics/srgb_to_linear.h"
// ...

TORCHSCIENCE_CPU_POINTWISE_UNARY(graphics, false, srgb_to_linear, x)
```

The same `macros/cpu/pointwise.h` is reused. No new macro definitions needed.

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

| Type | File | Shape Behavior | Examples |
|------|------|---------------|----------|
| Pointwise | `pointwise.h` | Element-wise with broadcasting | `gamma`, `beta`, `binomial_coefficient` |
| Reduction | `reduction.h` | Reduces dimensions | `kurtosis`, `kullback_leibler_divergence` |
| Creation | `creation.h` | Creates tensors from parameters | `rectangular_window`, `pink_noise` |
| Identity | `identity.h` | Preserves shape exactly | `srgb_to_hsv` |

Fixed, batched, and dynamic operator types are deferred until enough operators exist to establish a stable pattern.

### What Does Not Change

- **Kernel headers** (`csrc/kernel/`) — untouched, these are the actual implementations
- **Schema definitions** (`csrc/torchscience.cpp`) — untouched
- **Python wrappers** (`src/torchscience/<module>/`) — untouched
- **Registration file locations** (`cpu/special_functions.h`, etc.) — same paths, updated contents
- **Generated function signatures** — the macros produce the same forward/backward/backward_backward functions with the same dispatch registrations
