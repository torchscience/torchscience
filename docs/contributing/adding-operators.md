# Adding New Operators

This guide explains how to add a new pointwise operator to torchscience using the X-macro infrastructure.

## Overview

The X-macro system automates operator registration across all backends (CPU, Meta, Autograd, Autocast, Sparse, Quantized). Adding a new operator requires:

1. Implement the kernel (forward, backward, backward_backward)
2. Create a traits struct
3. Add ONE line to the `.def` file

## Step-by-Step Guide

### 1. Implement the Kernel

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

### 2. Create the Traits Struct

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

### 3. Add to the X-Macro Definition

Edit `src/torchscience/csrc/operators/<category>.def`:

```cpp
#ifndef TORCHSCIENCE_<CATEGORY>
#define TORCHSCIENCE_<CATEGORY>(X) \
    X(existing_op,  1, torchscience::impl::<category>::ExistingImpl) \
    X(my_operator,  1, torchscience::impl::<category>::MyOperatorImpl)
//    ^name         ^arity  ^fully qualified traits type
#endif
```

**That's it!** The X-macro system automatically:
- Generates schema definitions in `torchscience.cpp`
- Registers the CPU implementation
- Registers the Meta implementation (shape inference)
- Registers the Autograd wrapper (gradient support)
- Registers the Autocast wrapper (mixed precision)
- Registers all Sparse backends (COO/CSR, CPU/CUDA)
- Registers all Quantized backends (CPU/CUDA)

## Arity Reference

| Arity | Type | Example |
|-------|------|---------|
| 1 | Unary | `gamma(z)` |
| 2 | Binary | `chebyshev_polynomial_t(v, z)` |
| 3 | Ternary | `incomplete_beta(z, a, b)` |
| 4 | Quaternary | `hypergeometric_2_f_1(a, b, c, z)` |

## File Structure

```
src/torchscience/csrc/
├── core/
│   ├── pointwise_registration.h  # Arity-based registration templates
│   └── schema_generation.h       # Schema string generators
├── operators/
│   └── special_functions.def     # X-macro definitions
├── impl/
│   └── special_functions/
│       ├── gamma.h               # Kernel implementation
│       └── gamma_traits.h        # Traits struct
├── cpu/
│   └── special_functions.h       # CPU backend registration
├── meta/
│   └── special_functions.h       # Meta backend registration
├── autograd/
│   └── special_functions.h       # Autograd backend registration
└── autocast/
    └── special_functions.h       # Autocast backend registration
```

## Testing

After adding your operator, create tests in `tests/torchscience/<category>/test__<operator_name>.py`.

Run the test suite:
```bash
pytest tests/torchscience/<category>/test__<operator_name>.py -v
```
