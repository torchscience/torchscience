# Adding New Operators

This guide explains how to add new operators to torchscience using the X-macro infrastructure.

## Overview

The X-macro system automates operator registration across all backends (CPU, Meta, Autograd, Autocast, Sparse, Quantized). Adding a new operator requires:

1. Implement the kernel (forward, backward, backward_backward)
2. Create a traits struct
3. Add ONE line to the appropriate `.def` file

Torchscience supports several operator categories, each with its own signature pattern:

| Category | Signature Pattern | Example |
|----------|------------------|---------|
| **Pointwise** | `(Tensor input, ...)` with arity 1-4 | `gamma(z)`, `incomplete_beta(z, a, b)` |
| **Reductions** | `(Tensor input, int[]? dim, bool keepdim, <extra_args>)` | `kurtosis(input, dim, keepdim, fisher, bias)` |
| **Transforms** | `(Tensor input, int n, int dim, <extra_args>)` | `hilbert_transform(input, n, dim, padding_mode, ...)` |
| **Pairwise Distance** | `(Tensor x, Tensor y, <extra_args>)` | `minkowski_distance(x, y, p, weight)` |
| **Graphics** | `(Tensor t1, ..., Tensor tN)` with N inputs | `cook_torrance(t1, t2, t3, t4, t5)` |

## Pointwise Operators

Pointwise operators apply element-wise transformations to tensors. They support arities from 1 (unary) to 4 (quaternary).

### X-Macro Format

```cpp
X(name, arity, impl_type)
```

| Field | Description |
|-------|-------------|
| `name` | Operator name (e.g., `gamma`, `incomplete_beta`) |
| `arity` | Number of tensor inputs (1-4) |
| `impl_type` | Fully qualified traits struct type |

### Arity Reference

| Arity | Type | Example |
|-------|------|---------|
| 1 | Unary | `gamma(z)` |
| 2 | Binary | `chebyshev_polynomial_t(v, z)` |
| 3 | Ternary | `incomplete_beta(z, a, b)` |
| 4 | Quaternary | `hypergeometric_2_f_1(a, b, c, z)` |

### Adding a Pointwise Operator

#### 1. Implement the Kernel

Create the kernel implementation in `src/torchscience/csrc/impl/<category>/<operator_name>.h`:

```cpp
#pragma once

#include <ATen/ATen.h>
#include <cmath>

namespace torchscience::impl::<category> {

template<typename scalar_t>
inline scalar_t my_operator_forward(scalar_t x) {
    // Your forward implementation
    return std::sin(x);
}

template<typename scalar_t>
inline scalar_t my_operator_backward(scalar_t grad_output, scalar_t x) {
    // Gradient w.r.t. input
    return grad_output * std::cos(x);
}

template<typename scalar_t>
inline std::tuple<scalar_t, scalar_t> my_operator_backward_backward(
    scalar_t gg_x,
    scalar_t grad_output,
    scalar_t x
) {
    // Second-order gradients
    scalar_t grad_grad_output = gg_x * std::cos(x);
    scalar_t grad_x = -gg_x * grad_output * std::sin(x);
    return {grad_grad_output, grad_x};
}

}  // namespace torchscience::impl::<category>
```

#### 2. Create the Traits Struct

Create `src/torchscience/csrc/impl/<category>/<operator_name>_traits.h`:

```cpp
#pragma once

#include "<operator_name>.h"
#include <ATen/ATen.h>

namespace torchscience::impl::<category> {

struct MyOperatorImpl {
    // Dispatch forward to CPU kernel
    static at::Tensor dispatch_forward(const at::Tensor& input) {
        return input.clone();  // Placeholder - actual implementation
    }

    // Dispatch backward
    static at::Tensor dispatch_backward(
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        // Call the backward schema
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::my_operator_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(grad_output, input);
    }

    // Dispatch backward_backward
    static std::tuple<at::Tensor, at::Tensor> dispatch_backward_backward(
        const at::Tensor& gg_input,
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::my_operator_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(gg_input, grad_output, input);
    }
};

}  // namespace torchscience::impl::<category>
```

#### 3. Add to the X-Macro Definition

Edit `src/torchscience/csrc/operators/special_functions.def`:

```cpp
#ifndef TORCHSCIENCE_SPECIAL_FUNCTIONS
#define TORCHSCIENCE_SPECIAL_FUNCTIONS(X) \
    X(gamma,                    1, torchscience::impl::special_functions::GammaImpl) \
    X(my_operator,              1, torchscience::impl::special_functions::MyOperatorImpl)
//    ^name                     ^arity  ^fully qualified traits type
#endif
```

---

## Reduction Operators

Reduction operators aggregate tensor values along specified dimensions. They follow the signature pattern:

```
op(Tensor input, int[]? dim, bool keepdim, <extra_args>) -> Tensor
```

### X-Macro Format

```cpp
X(name, extra_args_schema, extra_args_count, impl_type)
```

| Field | Description |
|-------|-------------|
| `name` | Operator name (e.g., `kurtosis`) |
| `extra_args_schema` | Extra non-tensor arguments as schema string (e.g., `"bool fisher, bool bias"`) |
| `extra_args_count` | Number of extra arguments |
| `impl_type` | Fully qualified traits struct type |

### Generated Schemas

The X-macro system generates three schemas for each reduction operator:

- **Forward**: `op(Tensor input, int[]? dim, bool keepdim, <extra_args>) -> Tensor`
- **Backward**: `op_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim, <extra_args>) -> Tensor`
- **Backward-backward**: `op_backward_backward(Tensor grad_grad_input, Tensor grad_output, Tensor input, int[]? dim, bool keepdim, <extra_args>) -> (Tensor, Tensor)`

### Adding a Reduction Operator

Edit `src/torchscience/csrc/operators/reductions.def`:

```cpp
#ifndef TORCHSCIENCE_REDUCTIONS
#define TORCHSCIENCE_REDUCTIONS(X) \
    X(kurtosis, "bool fisher, bool bias", 2, torchscience::impl::statistics::descriptive::KurtosisImpl) \
    X(my_reduction, "float alpha", 1, torchscience::impl::my_category::MyReductionImpl)
//    ^name         ^extra args schema   ^count  ^fully qualified traits type
#endif
```

---

## Transform Operators

Transform operators apply fixed-dimension transformations (like FFT-based operations). They follow the signature pattern:

```
op(Tensor input, int n=-1, int dim=-1, <extra_args>) -> Tensor
```

### X-Macro Format

```cpp
X(name, extra_args_schema, impl_type)
```

| Field | Description |
|-------|-------------|
| `name` | Operator name (e.g., `hilbert_transform`) |
| `extra_args_schema` | Extra parameters with optional defaults (e.g., `"int padding_mode=0, float padding_value=0.0"`) |
| `impl_type` | Fully qualified traits struct type |

### Generated Schemas

The X-macro system generates three schemas for each transform operator:

- **Forward**: `op(Tensor input, int n=-1, int dim=-1, <extra_args>) -> Tensor`
- **Backward**: `op_backward(Tensor grad_output, Tensor input, int n, int dim, <extra_args_no_defaults>) -> Tensor`
- **Backward-backward**: `op_backward_backward(Tensor grad_grad_input, Tensor grad_output, Tensor input, int n, int dim, <extra_args_no_defaults>) -> (Tensor, Tensor)`

Note: Default values are stripped from `extra_args` in backward schemas.

### Adding a Transform Operator

Edit `src/torchscience/csrc/operators/transforms.def`:

```cpp
#ifndef TORCHSCIENCE_TRANSFORMS
#define TORCHSCIENCE_TRANSFORMS(X) \
    X(hilbert_transform, "int padding_mode=0, float padding_value=0.0, Tensor? window=None", \
      torchscience::impl::integral_transform::HilbertTransformImpl) \
    X(my_transform, "bool normalize=true", \
      torchscience::impl::my_category::MyTransformImpl)
//    ^name         ^extra args with defaults     ^fully qualified traits type
#endif
```

---

## Pairwise Distance Operators

Pairwise distance operators compute distances between two tensors. They follow the signature pattern:

```
op(Tensor x, Tensor y, <extra_args>) -> Tensor
```

### X-Macro Format

```cpp
X(name, extra_args_schema, impl_type)
```

| Field | Description |
|-------|-------------|
| `name` | Operator name (e.g., `minkowski_distance`) |
| `extra_args_schema` | Extra parameters (e.g., `"float p, Tensor? weight"`) |
| `impl_type` | Fully qualified traits struct type |

### Generated Schemas

The X-macro system generates two schemas for each pairwise distance operator:

- **Forward**: `op(Tensor x, Tensor y, <extra_args>) -> Tensor`
- **Backward**: `op_backward(Tensor grad_output, Tensor x, Tensor y, <extra_args_no_defaults>, Tensor dist_output) -> (Tensor, Tensor, Tensor)`

Note: The backward includes the forward output (`dist_output`) for efficient gradient computation.

### Adding a Pairwise Distance Operator

Edit `src/torchscience/csrc/operators/distance.def`:

```cpp
#ifndef TORCHSCIENCE_DISTANCES
#define TORCHSCIENCE_DISTANCES(X) \
    X(minkowski_distance, "float p, Tensor? weight", \
      torchscience::impl::distance::MinkowskiDistanceImpl) \
    X(my_distance, "float epsilon=1e-8", \
      torchscience::impl::distance::MyDistanceImpl)
//    ^name        ^extra args schema      ^fully qualified traits type
#endif
```

---

## Graphics Operators

Graphics operators handle multi-input tensor operations common in computer graphics (shading, lighting, etc.). They follow the signature pattern:

```
op(Tensor t1, Tensor t2, ..., Tensor tN) -> Tensor
```

### X-Macro Format

```cpp
X(name, input_count, impl_type)
```

| Field | Description |
|-------|-------------|
| `name` | Operator name (e.g., `cook_torrance`) |
| `input_count` | Number of input tensors (e.g., 5 for cook_torrance) |
| `impl_type` | Fully qualified traits struct type |

### Generated Schemas

The X-macro system generates three schemas for each graphics operator:

- **Forward**: `op(Tensor t1, ..., Tensor tN) -> Tensor`
- **Backward**: `op_backward(Tensor grad_output, Tensor t1, ..., Tensor tN) -> (Tensor, ..., Tensor)` (N outputs)
- **Backward-backward**: `op_backward_backward(Tensor gg_t1, ..., Tensor gg_tN, Tensor grad_output, Tensor t1, ..., Tensor tN) -> (Tensor, ..., Tensor)` (N+1 outputs)

### Adding a Graphics Operator

Edit `src/torchscience/csrc/operators/graphics.def`:

```cpp
#ifndef TORCHSCIENCE_GRAPHICS
#define TORCHSCIENCE_GRAPHICS(X) \
    X(cook_torrance, 5, torchscience::impl::graphics::shading::CookTorranceImpl) \
    X(my_shader, 3, torchscience::impl::graphics::shading::MyShaderImpl)
//    ^name      ^input count  ^fully qualified traits type
#endif
```

---

## File Structure

```
src/torchscience/csrc/
├── core/
│   ├── pointwise_registration.h  # Arity-based registration templates
│   ├── schema_generation.h       # Schema string generators (pointwise)
│   ├── reduction_schema.h        # Schema generators for reductions
│   ├── transform_schema.h        # Schema generators for transforms
│   ├── pairwise_schema.h         # Schema generators for pairwise distance
│   └── graphics_schema.h         # Schema generators for graphics
├── operators/
│   ├── special_functions.def     # Pointwise operators X-macro definitions
│   ├── reductions.def            # Reduction operators X-macro definitions
│   ├── transforms.def            # Transform operators X-macro definitions
│   ├── distance.def              # Pairwise distance operators X-macro definitions
│   └── graphics.def              # Graphics operators X-macro definitions
├── impl/
│   ├── special_functions/
│   │   ├── gamma.h               # Kernel implementation
│   │   └── gamma_traits.h        # Traits struct
│   ├── statistics/
│   │   └── descriptive/
│   │       └── kurtosis_traits.h # Reduction traits
│   ├── integral_transform/
│   │   └── hilbert_transform_traits.h  # Transform traits
│   ├── distance/
│   │   └── minkowski_distance_traits.h # Distance traits
│   └── graphics/
│       └── shading/
│           └── cook_torrance_traits.h  # Graphics traits
├── cpu/
│   └── special_functions.h       # CPU backend registration
├── meta/
│   └── special_functions.h       # Meta backend registration
├── autograd/
│   └── special_functions.h       # Autograd backend registration
└── autocast/
    └── special_functions.h       # Autocast backend registration
```

## How It Works

The X-macro system automatically:
- Generates schema definitions in `torchscience.cpp`
- Registers the CPU implementation
- Registers the Meta implementation (shape inference)
- Registers the Autograd wrapper (gradient support)
- Registers the Autocast wrapper (mixed precision)
- Registers all Sparse backends (COO/CSR, CPU/CUDA)
- Registers all Quantized backends (CPU/CUDA)

## Testing

After adding your operator, create tests in `tests/torchscience/<category>/test__<operator_name>.py`.

Run the test suite:
```bash
pytest tests/torchscience/<category>/test__<operator_name>.py -v
```
