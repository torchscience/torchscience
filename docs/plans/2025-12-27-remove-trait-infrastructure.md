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

## Phase 2: Other Operator Categories

Same pattern as Phase 1. Each operator gets its own file with inline kernels.

**Note:** These operators already have per-operator files in cpu/, meta/, autograd/, autocast/.
The existing code just needs to be verified for consistency with the Phase 1 pattern (no helpers,
explicit registration). Sparse and quantized backends have stub implementations.

---

### Task 2.1: Verify kurtosis (statistics/descriptive)

**Files to check:**
- `cpu/statistics/descriptive/kurtosis.h`
- `meta/statistics/descriptive/kurtosis.h`
- `autograd/statistics/descriptive/kurtosis.h`
- `autocast/statistics/descriptive/kurtosis.h`

**Verification:**
```bash
.venv/bin/python -m pytest tests/torchscience/statistics/descriptive/test__kurtosis.py -v
```

---

### Task 2.2: Verify histogram (statistics/descriptive)

**Files to check:**
- `cpu/statistics/descriptive/histogram.h`
- `meta/statistics/descriptive/histogram.h`

**Note:** histogram doesn't have backward (non-differentiable), so no autograd/autocast needed.

**Verification:**
```bash
.venv/bin/python -m pytest tests/torchscience/statistics/descriptive/test__histogram.py -v
```

---

### Task 2.3: Verify hilbert_transform (integral_transform)

**Files to check:**
- `cpu/integral_transform/hilbert_transform.h`
- `meta/integral_transform/hilbert_transform.h`
- `autograd/integral_transform/hilbert_transform.h`
- `autocast/integral_transform/hilbert_transform.h`

**Verification:**
```bash
.venv/bin/python -m pytest tests/torchscience/integral_transform/ -v
```

---

### Task 2.4: Verify inverse_hilbert_transform (integral_transform)

**Files to check:**
- `cpu/integral_transform/inverse_hilbert_transform.h`
- `meta/integral_transform/inverse_hilbert_transform.h`
- `autograd/integral_transform/inverse_hilbert_transform.h`
- `autocast/integral_transform/inverse_hilbert_transform.h`

**Verification:**
```bash
.venv/bin/python -m pytest tests/torchscience/integral_transform/ -v
```

---

### Task 2.5: Verify minkowski_distance (distance)

**Files to check:**
- `cpu/distance/minkowski_distance.h`
- `meta/distance/minkowski_distance.h`
- `autograd/distance/minkowski_distance.h`

**Verification:**
```bash
.venv/bin/python -m pytest tests/torchscience/distance/ -v
```

---

### Task 2.6: Verify cook_torrance (graphics/shading)

**Files to check:**
- `cpu/graphics/shading/cook_torrance.h`
- `meta/graphics/shading/cook_torrance.h`
- `autograd/graphics/shading/cook_torrance.h`

**Verification:**
```bash
.venv/bin/python -m pytest tests/torchscience/graphics/shading/ -v
```

---

### Task 2.7: Verify rosenbrock (optimization/test_functions)

**Files to check:**
- `cpu/optimization/test_functions.h`
- `meta/optimization/test_functions.h`
- `autograd/optimization/test_functions.h`

**Verification:**
```bash
.venv/bin/python -m pytest tests/torchscience/optimization/ -v
```

---

### Task 2.8: Verify signal_processing (filter, waveforms, windows)

**Files to check:**
- `cpu/signal_processing/filter.h`
- `autograd/signal_processing/filter.h`
- `meta/signal_processing/filter.h`
- `autocast/signal_processing/filter.h`
- `composite/signal_processing/waveform.h`
- `composite/signal_processing/window_functions.h`

**Verification:**
```bash
.venv/bin/python -m pytest tests/torchscience/signal_processing/ -v
```

---

### Task 2.9: Update sparse/quantized backends

The sparse (COO, CSR) and quantized backends have stub implementations that throw errors.
These are already in per-operator files. Just verify includes are correct.

**Files:**
- `sparse/coo/cpu/special_functions.h` → should include per-operator files
- `sparse/csr/cpu/special_functions.h` → should include per-operator files
- `quantized/cpu/special_functions.h` → should include per-operator files

---

### Task 2.10: Run full test suite

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All tests pass (or skip for CUDA/sparse/quantized as before).

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
