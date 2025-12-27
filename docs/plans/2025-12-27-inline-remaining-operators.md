# Inline Remaining Operators - Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Inline all remaining `impl/` code into per-operator files, then delete `impl/` directory entirely.

**Pattern:** Same as Phase 1 special_functions - per-operator files with inline kernels, no helpers, explicit registration.

---

## What Gets Inlined

| Operator | impl/ Lines | Target Files |
|----------|-------------|--------------|
| minkowski_distance | ~235 | cpu, meta, autograd |
| rosenbrock | ~31 | cpu, meta, autograd |
| kurtosis | ~630 | cpu, meta, autograd, autocast |
| histogram | ~535 | cpu, meta, autograd, autocast |
| hilbert_transform | ~700 | cpu, meta, autograd, autocast |
| inverse_hilbert_transform | ~255 | cpu, meta, autograd, autocast |
| cook_torrance | ~1,213 | cpu, meta, autograd |
| butterworth_filter | ~1,178 | cpu, meta, autograd, autocast |
| **Total** | **~4,778** | |

---

## File Structure (after inlining)

```
cpu/
  distance/minkowski_distance.h          # inline kernels + registration
  graphics/shading/cook_torrance.h       # inline kernels + registration
  integral_transform/hilbert_transform.h
  integral_transform/inverse_hilbert_transform.h
  optimization/test_functions.h          # rosenbrock
  signal_processing/filter.h             # butterworth
  statistics/descriptive/kurtosis.h
  statistics/descriptive/histogram.h

meta/
  (same structure - shape inference only)

autograd/
  (same structure - autograd wrappers)

autocast/
  (same structure - dtype casting)

impl/  <- DELETED
```

---

## Template for CPU Files

```cpp
#pragma once

#include <cmath>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

namespace torchscience::cpu::<category> {

namespace {

// Inline kernel implementations (moved from impl/)
template<typename T>
T operator_kernel(...) {
    // math here
}

}  // anonymous namespace

inline at::Tensor operator_forward(...) {
    // TensorIterator setup
    // AT_DISPATCH_FLOATING_TYPES_AND2
    // kernel call
}

inline at::Tensor operator_backward(...) { /* ... */ }

inline std::tuple<...> operator_backward_backward(...) { /* ... */ }

}  // namespace

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("operator", torchscience::cpu::<category>::operator_forward);
    m.impl("operator_backward", ...);
    m.impl("operator_backward_backward", ...);
}
```

---

## Special Considerations

1. **histogram** - No backward pass but still needs autograd (pass-through) and autocast (dtype handling) for proper dispatch.

2. **hilbert_transform / inverse_hilbert_transform** - Uses FFT internally. Padding/windowing helpers inline into CPU file.

3. **cook_torrance** - Largest operator (1,213 lines). backward_backward alone is 904 lines.

4. **butterworth_analog_bandpass_filter** - Complex analog filter math. Three impl files merge into one.

5. **Sparse/Quantized backends** - Already have stub implementations, no changes needed.

---

## Tasks

### Task 1: Inline minkowski_distance
- Read `impl/distance/minkowski_distance.h` and `minkowski_distance_backward.h`
- Inline into `cpu/distance/minkowski_distance.h`
- Update `meta/distance/minkowski_distance.h` (remove impl include if any)
- Update `autograd/distance/minkowski_distance.h` (remove impl include if any)
- Delete impl files
- Run tests: `pytest tests/torchscience/distance/ -v`

### Task 2: Inline rosenbrock
- Read `impl/optimization/test_functions.h`
- Inline into `cpu/optimization/test_functions.h`
- Update meta/autograd files
- Delete impl files
- Run tests: `pytest tests/torchscience/optimization/ -v`

### Task 3: Inline kurtosis
- Read `impl/statistics/descriptive/kurtosis*.h` (3 files)
- Inline into `cpu/statistics/descriptive/kurtosis.h`
- Update meta/autograd/autocast files
- Delete impl files
- Run tests: `pytest tests/torchscience/statistics/ -v`

### Task 4: Inline histogram
- Read `impl/statistics/descriptive/histogram.h`
- Inline into `cpu/statistics/descriptive/histogram.h`
- Update meta file, add autograd/autocast pass-through
- Delete impl files
- Run tests: `pytest tests/torchscience/statistics/ -v`

### Task 5: Inline hilbert_transform
- Read `impl/integral_transform/hilbert_transform*.h` (3 files)
- Inline into `cpu/integral_transform/hilbert_transform.h`
- Update meta/autograd/autocast files
- Delete impl files
- Run tests: `pytest tests/torchscience/integral_transform/ -v`

### Task 6: Inline inverse_hilbert_transform
- Read `impl/integral_transform/inverse_hilbert_transform*.h` (3 files)
- Inline into `cpu/integral_transform/inverse_hilbert_transform.h`
- Update meta/autograd/autocast files
- Delete impl files
- Run tests: `pytest tests/torchscience/integral_transform/ -v`

### Task 7: Inline cook_torrance
- Read `impl/graphics/shading/cook_torrance*.h` (3 files)
- Inline into `cpu/graphics/shading/cook_torrance.h`
- Update meta/autograd files
- Delete impl files
- Run tests: `pytest tests/torchscience/graphics/ -v`

### Task 8: Inline butterworth_analog_bandpass_filter
- Read `impl/signal_processing/filter/butterworth*.h` (3 files)
- Inline into `cpu/signal_processing/filter.h`
- Update meta/autograd/autocast files
- Delete impl files
- Run tests: `pytest tests/torchscience/signal_processing/ -v`

### Task 9: Final cleanup
- Delete empty `impl/` directory
- Run full test suite
- Commit all changes

---

## Success Criteria

- All tests pass (881+ passed)
- `impl/` directory deleted
- No `#include "impl/..."` anywhere in codebase
- Each operator is self-contained in its own file
