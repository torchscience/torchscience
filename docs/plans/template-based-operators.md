# Implementation Plan: Replace Macro-Based Operators with C++20 Templates

## Overview

Replace the current preprocessor macro system (`macros.h` files totaling ~5,800 lines across 10 files) with a type-safe C++20 template-based approach. Templates handle all operator logic; minimal macros handle only string concatenation (which C++ cannot do without preprocessor).

## Design Decision: Hybrid Template + Minimal Macro

Due to the [constexpr string firewall](https://quuxplusone.github.io/blog/2023/09/08/constexpr-string-firewall/), compile-time string concatenation cannot reliably cross into runtime. Therefore:

- **Templates:** Handle all operator logic (TensorIterator, dispatch, kernels)
- **Minimal Macros:** Handle only `#name` stringification and `"_backward"` concatenation

This gives us type-safety and debuggability for the complex logic while acknowledging that C++ genuinely requires macros for string concatenation.

## Current State

### Macro Files (10 total, ~5,800 lines)
- `cpu/macros.h` - CPU dispatch with TensorIterator
- `meta/macros.h` - Shape inference for Meta backend
- `autograd/macros.h` - Custom autograd::Function classes
- `autocast/macros.h` - Mixed precision casting
- `sparse/coo/cpu/macros.h` - Sparse COO CPU dispatch
- `sparse/coo/cuda/macros.h` - Sparse COO CUDA dispatch
- `sparse/csr/cpu/macros.h` - Sparse CSR CPU dispatch
- `sparse/csr/cuda/macros.h` - Sparse CSR CUDA dispatch
- `quantized/cpu/macros.h` - Quantized CPU dispatch
- `quantized/cuda/macros.h` - Quantized CUDA dispatch

### Current usage:
```cpp
// cpu/special_functions.h
#include "macros.h"
CPU_UNARY_OPERATOR(special_functions, gamma, z)
CPU_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)
```

## Target State

```cpp
// Templates handle all logic
template<typename ImplTraits>
struct CPUUnaryOperator {
    static at::Tensor forward(const at::Tensor& input) { /* ... */ }
    static at::Tensor backward(/* ... */) { /* ... */ }
    static std::tuple<at::Tensor, at::Tensor> backward_backward(/* ... */) { /* ... */ }

    static void register_all(
        torch::Library& m,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name
    ) {
        m.impl(name, &forward);
        m.impl(backward_name, &backward);
        m.impl(backward_backward_name, &backward_backward);
    }
};

// Minimal macro only for string concatenation
#define REGISTER_CPU_UNARY(module, name, Impl) \
    CPUUnaryOperator<Impl>::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

// Usage
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    REGISTER_CPU_UNARY(module, gamma, GammaImpl);
    REGISTER_CPU_BINARY(module, chebyshev_polynomial_t, ChebyshevTImpl);
}
```

---

## Implementation Tasks

### Task 1: Create CPU unary operator template
**File:** `src/torchscience/csrc/cpu/operators.h` (new, replaces `cpu/macros.h`)

```cpp
#pragma once

#include <tuple>
#include <ATen/Dispatch.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

namespace torchscience::cpu {

template<typename ImplTraits>
struct CPUUnaryOperator {
    static at::Tensor forward(const at::Tensor& input) {
        at::Tensor output;

        auto iter = at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "unary_forward",
            [&]() {
                at::native::cpu_kernel(iter, [](scalar_t x) -> scalar_t {
                    return ImplTraits::template forward<scalar_t>(x);
                });
            }
        );

        return iter.output();
    }

    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        at::Tensor grad_input;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_input)
            .add_const_input(grad_output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "unary_backward",
            [&]() {
                at::native::cpu_kernel(iter, [](scalar_t g, scalar_t x) -> scalar_t {
                    return ImplTraits::template backward<scalar_t>(g, x);
                });
            }
        );

        return iter.output();
    }

    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        const bool has_gg = grad_grad_input.defined();

        if (!has_gg) {
            return std::make_tuple(at::Tensor(), at::Tensor());
        }

        at::Tensor grad_grad_output;
        at::Tensor grad_input;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_grad_output)
            .add_output(grad_input)
            .add_const_input(grad_grad_input)
            .add_const_input(grad_output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "unary_backward_backward",
            [&]() {
                at::native::cpu_kernel_multiple_outputs(
                    iter,
                    [has_gg](scalar_t gg, scalar_t g, scalar_t x)
                        -> std::tuple<scalar_t, scalar_t> {
                        return ImplTraits::template backward_backward<scalar_t>(
                            gg, g, x, has_gg
                        );
                    }
                );
            }
        );

        return std::make_tuple(iter.output(0), iter.output(1));
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name
    ) {
        module.impl(name, &forward);
        module.impl(backward_name, &backward);
        module.impl(backward_backward_name, &backward_backward);
    }
};

// Minimal macro for string concatenation only
#define REGISTER_CPU_UNARY(module, name, Impl) \
    ::torchscience::cpu::CPUUnaryOperator<Impl>::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

}  // namespace torchscience::cpu
```

**Verification:** Compiles and can instantiate with a test ImplTraits.

---

### Task 2: Create CPU binary operator template
**File:** `src/torchscience/csrc/cpu/operators.h` (add to same file)

Same pattern as Task 1 with two inputs:
- `forward(input1, input2) -> output`
- `backward(grad_output, input1, input2) -> (grad1, grad2)`
- `backward_backward(gg1, gg2, grad_output, input1, input2) -> (grad_grad_output, grad1, grad2)`

```cpp
#define REGISTER_CPU_BINARY(module, name, Impl) \
    ::torchscience::cpu::CPUBinaryOperator<Impl>::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")
```

---

### Task 3: Create CPU ternary operator template
**File:** `src/torchscience/csrc/cpu/operators.h` (add to same file)

Extended to 3 inputs.

---

### Task 4: Create CPU quaternary operator template
**File:** `src/torchscience/csrc/cpu/operators.h` (add to same file)

Extended to 4 inputs.

---

### Task 5: Create Meta operator templates
**File:** `src/torchscience/csrc/meta/operators.h` (new, replaces `meta/macros.h`)

Meta operators compute output shapes only (no kernel dispatch):

```cpp
namespace torchscience::meta {

struct MetaUnaryOperator {
    static at::Tensor forward(const at::Tensor& input) {
        at::Tensor output;
        return at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build()
            .output();
    }

    static at::Tensor backward(const at::Tensor& grad_output, const at::Tensor& input) {
        // Similar shape inference
    }

    static std::tuple<at::Tensor, at::Tensor> backward_backward(/* ... */) {
        // Similar shape inference
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name
    );
};

#define REGISTER_META_UNARY(module, name) \
    ::torchscience::meta::MetaUnaryOperator::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

}  // namespace torchscience::meta
```

---

### Task 6: Create Autograd operator templates
**File:** `src/torchscience/csrc/autograd/operators.h` (new, replaces `autograd/macros.h`)

Wraps `torch::autograd::Function` classes:

```cpp
namespace torchscience::autograd {

template<typename ImplTraits>
struct AutogradUnaryOperator {
    class Backward : public torch::autograd::Function<Backward> {
    public:
        static std::vector<at::Tensor> forward(
            torch::autograd::AutogradContext* ctx,
            const at::Tensor& grad_output,
            const at::Tensor& input,
            bool input_requires_grad
        );

        static std::vector<at::Tensor> backward(
            torch::autograd::AutogradContext* ctx,
            const std::vector<at::Tensor>& grad_outputs
        );
    };

    class Forward : public torch::autograd::Function<Forward> {
    public:
        static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            const at::Tensor& input
        );

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            const torch::autograd::variable_list& grad_outputs
        );
    };

    static at::Tensor apply(const at::Tensor& input) {
        return Forward::apply(input);
    }

    static void register_all(torch::Library& module, const char* name) {
        module.impl(name, &apply);
    }
};

#define REGISTER_AUTOGRAD_UNARY(module, name, Impl) \
    ::torchscience::autograd::AutogradUnaryOperator<Impl>::register_all(module, #name)

}  // namespace torchscience::autograd
```

---

### Task 7: Create Autocast operator templates
**File:** `src/torchscience/csrc/autocast/operators.h` (new, replaces `autocast/macros.h`)

```cpp
namespace torchscience::autocast {

struct AutocastUnaryOperator {
    static at::Tensor forward(const at::Tensor& input, const char* name) {
        c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Autocast);

        at::ScalarType dtype = input.device().is_cpu()
            ? at::autocast::get_autocast_dtype(at::kCPU)
            : at::autocast::get_autocast_dtype(at::kCUDA);

        // Handle complex types...

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(name, "")
            .typed<at::Tensor(const at::Tensor&)>()
            .call(at::autocast::cached_cast(dtype, input));
    }

    static void register_all(torch::Library& module, const char* name);
};

#define REGISTER_AUTOCAST_UNARY(module, name) \
    ::torchscience::autocast::AutocastUnaryOperator::register_all(module, "torchscience::" #name)

}  // namespace torchscience::autocast
```

---

### Task 8: Create Sparse COO CPU operator templates
**File:** `src/torchscience/csrc/sparse/coo/cpu/operators.h` (new, replaces `sparse/coo/cpu/macros.h`)

---

### Task 9: Create Sparse COO CUDA operator templates
**File:** `src/torchscience/csrc/sparse/coo/cuda/operators.h` (new, replaces `sparse/coo/cuda/macros.h`)

---

### Task 10: Create Sparse CSR CPU operator templates
**File:** `src/torchscience/csrc/sparse/csr/cpu/operators.h` (new, replaces `sparse/csr/cpu/macros.h`)

---

### Task 11: Create Sparse CSR CUDA operator templates
**File:** `src/torchscience/csrc/sparse/csr/cuda/operators.h` (new, replaces `sparse/csr/cuda/macros.h`)

---

### Task 12: Create Quantized CPU operator templates
**File:** `src/torchscience/csrc/quantized/cpu/operators.h` (new, replaces `quantized/cpu/macros.h`)

---

### Task 13: Create Quantized CUDA operator templates
**File:** `src/torchscience/csrc/quantized/cuda/operators.h` (new, replaces `quantized/cuda/macros.h`)

---

### Task 14: Create ImplTraits for gamma
**File:** `src/torchscience/csrc/impl/special_functions/gamma.h` (modify - add at end)

```cpp
namespace torchscience::impl::special_functions {

struct GammaImpl {
    template<typename T>
    static T forward(T z) {
        return gamma(z);
    }

    template<typename T>
    static T backward(T grad, T z) {
        return gamma_backward(grad, z);
    }

    template<typename T>
    static std::tuple<T, T> backward_backward(T gg, T grad, T z, bool has_gg) {
        return gamma_backward_backward(gg, grad, z, has_gg);
    }
};

}  // namespace torchscience::impl::special_functions
```

---

### Task 15: Migrate gamma registration
**File:** `src/torchscience/csrc/torchscience.cpp` (modify)

```cpp
#include "cpu/operators.h"
#include "meta/operators.h"
#include "autograd/operators.h"
#include "autocast/operators.h"
#include "impl/special_functions/gamma.h"

using torchscience::impl::special_functions::GammaImpl;

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    REGISTER_CPU_UNARY(module, gamma, GammaImpl);
}

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    REGISTER_META_UNARY(module, gamma);
}

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    REGISTER_AUTOGRAD_UNARY(module, gamma, GammaImpl);
}

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
    REGISTER_AUTOCAST_UNARY(module, gamma);
}
```

---

### Task 16: Verify gamma works
**Test:**
```bash
python -c "
import torch
import torchscience

x = torch.randn(10, requires_grad=True)
y = torchscience.gamma(x)
y.sum().backward()
print('Forward + backward: OK')

# Test double backward
x2 = torch.randn(5, requires_grad=True)
y2 = torchscience.gamma(x2)
g = torch.autograd.grad(y2.sum(), x2, create_graph=True)[0]
g.sum().backward()
print('Double backward: OK')
"
```

---

### Task 17: Create ImplTraits for remaining operators
- `ChebyshevTImpl` (binary)
- `IncompleteBetaImpl` (ternary)
- `Hypergeometric2F1Impl` (quaternary)

---

### Task 18: Migrate remaining operator registrations

---

### Task 19: Remove legacy macro files
Delete all `macros.h` files once migration is complete:
- `cpu/macros.h`
- `meta/macros.h`
- `autograd/macros.h`
- `autocast/macros.h`
- `sparse/coo/cpu/macros.h`
- `sparse/coo/cuda/macros.h`
- `sparse/csr/cpu/macros.h`
- `sparse/csr/cuda/macros.h`
- `quantized/cpu/macros.h`
- `quantized/cuda/macros.h`

---

### Task 20: Update documentation

---

## File Structure After Migration

```
src/torchscience/csrc/
├── cpu/
│   └── operators.h              # CPUUnaryOperator, CPUBinaryOperator, etc.
├── meta/
│   └── operators.h              # MetaUnaryOperator, MetaBinaryOperator, etc.
├── autograd/
│   └── operators.h              # AutogradUnaryOperator, etc.
├── autocast/
│   └── operators.h              # AutocastUnaryOperator, etc.
├── sparse/
│   ├── coo/
│   │   ├── cpu/
│   │   │   └── operators.h
│   │   └── cuda/
│   │       └── operators.h
│   └── csr/
│       ├── cpu/
│       │   └── operators.h
│       └── cuda/
│           └── operators.h
├── quantized/
│   ├── cpu/
│   │   └── operators.h
│   └── cuda/
│       └── operators.h
├── impl/
│   └── special_functions/
│       ├── gamma.h              # Math impl + GammaImpl traits
│       ├── chebyshev_polynomial_t.h
│       └── ...
└── torchscience.cpp             # All registrations using REGISTER_* macros
```

---

## Benefits

1. **Type Safety:** Template logic is type-checked at compile time
2. **Debuggability:** Step through template code in debugger (no macro expansion)
3. **IDE Support:** Autocomplete, go-to-definition work on templates
4. **Reduced Duplication:** ~5,800 lines -> ~800 lines of templates + ~50 lines of macros
5. **Clear Separation:** Templates = logic, Macros = only string concatenation

## Macro Comparison

| Before | After |
|--------|-------|
| `CPU_UNARY_OPERATOR(ns, name, arg)` (~180 lines) | `REGISTER_CPU_UNARY(m, name, Impl)` (2 lines) |
| Macro contains all logic | Macro only does `#name "_backward"` |
| Hard to debug | Templates are debuggable |
| No type checking | Full type checking |

---

## Verification Checklist

- [ ] CPUUnaryOperator compiles
- [ ] CPUBinaryOperator compiles
- [ ] CPUTernaryOperator compiles
- [ ] CPUQuaternaryOperator compiles
- [ ] MetaOperator compiles
- [ ] AutogradOperator compiles
- [ ] AutocastOperator compiles
- [ ] Gamma operator passes all existing tests
- [ ] All operators migrated
- [ ] Legacy macro files removed
