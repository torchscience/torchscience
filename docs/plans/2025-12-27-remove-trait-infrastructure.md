# Remove All Infrastructure - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Delete all abstraction layers. Each operator is a self-contained file with inline kernels and direct registration. No `core/`, no `impl/`, no templates, no traits, no X-macros.

**End state:** One file per operator per backend. The math is inline. The registration is explicit.

---

## Design Decisions (from Phase 1 implementation)

1. **Per-operator files**: Each operator gets its own header file
   - `cpu/special_functions/gamma.h` - not consolidated `cpu/special_functions.h`
   - Main `special_functions.h` just includes the individual operator files

2. **No helpers**: Each operator is fully self-contained
   - No `unary_meta()`, `binary_meta()` helper functions
   - Explicit TensorIterator setup in each operator function
   - Copy-paste is preferred over abstraction

3. **Inline kernels**: All math is in the operator file
   - `gamma_kernel()`, `digamma_kernel()`, etc. in anonymous namespace
   - No separate impl/ directory

---

## What Gets Deleted

### Entire directories:
- `src/torchscience/csrc/core/` (996 lines) ✅ DELETED
- `src/torchscience/csrc/impl/` (~3,000 lines) ✅ DELETED
- `src/torchscience/csrc/operators/` (103 lines) ✅ DELETED

### Template files:
- `cpu/operators.h` (628 lines) ✅ DELETED
- `meta/operators.h` (435 lines) ✅ DELETED
- `autograd/operators.h` (530 lines) ✅ DELETED
- `autocast/operators.h` (~200 lines) ✅ DELETED

**Total deleted: ~6,000 lines**

---

## Phase 1: Special Functions ✅ COMPLETE

**Actual implementation uses per-operator files:**

### File Structure (implemented)

```
cpu/
  special_functions.h          # Just includes:
  special_functions/
    gamma.h                    # Inline kernels + registration
    chebyshev_polynomial_t.h   # Inline kernels + registration
    incomplete_beta.h          # Stub + registration
    hypergeometric_2_f_1.h     # Stub + registration

meta/
  special_functions.h          # Just includes:
  special_functions/
    gamma.h                    # Shape inference + registration
    chebyshev_polynomial_t.h
    incomplete_beta.h
    hypergeometric_2_f_1.h

autograd/
  special_functions.h          # Just includes:
  special_functions/
    gamma.h                    # Autograd classes + registration
    chebyshev_polynomial_t.h
    incomplete_beta.h
    hypergeometric_2_f_1.h

autocast/
  special_functions.h          # Just includes:
  special_functions/
    gamma.h                    # Cast + redispatch + registration
    chebyshev_polynomial_t.h
    incomplete_beta.h
    hypergeometric_2_f_1.h
```

### Example: cpu/special_functions/gamma.h

```cpp
#pragma once

#include <cmath>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

namespace torchscience::cpu {

namespace {

template<typename T>
T gamma_kernel(T z) {
    // Lanczos approximation - full implementation inline
    constexpr T g = T(7);
    constexpr T coefficients[] = { /* ... */ };
    // ... full algorithm
}

template<typename T>
T digamma_kernel(T x) { /* ... */ }

template<typename T>
T trigamma_kernel(T x) { /* ... */ }

}  // anonymous namespace

inline at::Tensor gamma_forward(const at::Tensor& input) {
    at::Tensor output;
    auto iter = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, iter.common_dtype(), "gamma_cpu", [&] {
            at::native::cpu_kernel(iter, [](scalar_t x) -> scalar_t {
                return gamma_kernel(x);
            });
        });

    return iter.output();
}

inline at::Tensor gamma_backward(const at::Tensor& grad, const at::Tensor& input) {
    // Full implementation inline - no shared code
}

inline std::tuple<at::Tensor, at::Tensor> gamma_backward_backward(
    const at::Tensor& gg, const at::Tensor& grad, const at::Tensor& input
) {
    // Full implementation inline - no shared code
}

}  // namespace torchscience::cpu

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("gamma", torchscience::cpu::gamma_forward);
    m.impl("gamma_backward", torchscience::cpu::gamma_backward);
    m.impl("gamma_backward_backward", torchscience::cpu::gamma_backward_backward);
}
```

### Tasks 1.1-1.7: ✅ COMPLETE

All tasks completed. Tests pass (384 passed, 92 skipped).

---

## Phase 2: Other Operator Categories ✅ COMPLETE

Same pattern as Phase 1. Each operator gets its own file with inline kernels.

**Completed 2025-12-27:** All impl/ files deleted, kernels inlined into per-operator files.

---

### Task 2.1: Inline kurtosis ✅ COMPLETE

Inlined kernel code from `impl/statistics/descriptive/kurtosis*.h` into `cpu/statistics/descriptive/kurtosis.h`.

---

### Task 2.2: Inline histogram ✅ COMPLETE

Inlined kernel code from `impl/statistics/descriptive/histogram.h` into `cpu/statistics/descriptive/histogram.h`.

---

### Task 2.3: Inline hilbert_transform ✅ COMPLETE

Inlined kernel code from `impl/integral_transform/hilbert_transform*.h` into `cpu/integral_transform/hilbert_transform.h`.

---

### Task 2.4: Inline inverse_hilbert_transform ✅ COMPLETE

Inlined kernel code from `impl/integral_transform/inverse_hilbert_transform*.h` into `cpu/integral_transform/inverse_hilbert_transform.h`.

**Note:** Renamed duplicate helper functions (`apply_padding`, `apply_window`, `adjust_backward_gradient_size`) with `inverse_` prefix to avoid ODR violations when both files are included.

---

### Task 2.5: Inline minkowski_distance ✅ COMPLETE

Inlined kernel code from `impl/distance/minkowski_distance*.h` into `cpu/distance/minkowski_distance.h`.

---

### Task 2.6: Inline cook_torrance ✅ COMPLETE

Inlined kernel code from `impl/graphics/shading/cook_torrance*.h` into `cpu/graphics/shading/cook_torrance.h`.

---

### Task 2.7: Inline rosenbrock ✅ COMPLETE

Inlined `check_rosenbrock_input` from `impl/optimization/test_functions.h` into `composite/optimization/test_functions.h`.

---

### Task 2.8: Inline butterworth_filter ✅ COMPLETE

Inlined kernel code from `impl/signal_processing/filter/butterworth_analog_bandpass_filter*.h` into `cpu/signal_processing/filter.h`.

Also:
- Inlined `RectangularWindowTraits` into `composite/signal_processing/window_functions.h`
- Inlined `creation_common.h` utilities into `cpu/creation_operators.h` and `meta/creation_operators.h`

---

### Task 2.9: Fix build errors ✅ COMPLETE

Fixed various build errors during inlining:
- Fixed `constexpr` issues with Half/BFloat16 types in `gamma.h` (use `double` constants with `static_cast`)
- Updated autocast API to new PyTorch `promote_type(dtype, device_type, tensors...)` signature
- Fixed schema mismatches in `torchscience.cpp`:
  - `minkowski_distance` (added weight parameter)
  - `sine_wave`, `rectangular_window` (corrected parameter types)
  - `kurtosis` (added fisher, bias parameters)
  - `hilbert_transform`/`inverse_hilbert_transform` (added all parameters)

---

### Task 2.10: Run full test suite ✅ COMPLETE

```bash
.venv/bin/python -m pytest tests/ -v
```

**Results:** 549 passed, 96 skipped, 2 xfailed

Pre-existing failures in `incomplete_beta` tests (argument order mismatch between Python wrapper and C++ schema) are unrelated to this refactoring.

---

## Summary

| What | Lines |
|------|-------|
| Deleted | ~6,000 |
| Added | ~2,000 |
| **Net reduction** | **~4,000** |

No more:
- `core/` directory
- `impl/` directory
- `operators/` directory
- X-macros
- Traits structs
- Template-based registration
- Dispatch helpers

Just functions that do the thing and register themselves.
