# Implementation Plan: Creation Operator Macros

## Overview

Create a macro system for creation/factory operators that generate tensors from parameters (not from input tensors). Examples include window functions, waveforms, noise generators, and structured matrices.

## Design Decisions

| Aspect | Decision |
|--------|----------|
| **API Style** | Match PyTorch - fixed-rank outputs, no `batch_shape` |
| **Dispatch** | Separate CPU/CUDA with custom kernels |
| **Sparse** | Separate CPU/CUDA macros for COO and CSR |
| **Quantized** | Separate CPU/CUDA macros |
| **Standard Params** | `dtype`, `layout`, `device`, `requires_grad` |
| **Stochastic Ops** | Add `generator` parameter |

## Reference Implementations

- `rectangular_window`: `src/torchscience/csrc/composite/signal_processing/window_functions.h`
- `sine_wave`: `src/torchscience/csrc/composite/signal_processing/waveform.h`
- Testing base: `src/torchscience/testing/creation_op_base.py`

---

## Tasks

### Task 1: Create CPU creation macro

**File:** `src/torchscience/csrc/cpu/creation_macros.h`

**Macro Signature:**
```cpp
#define CPU_CREATION_OPERATOR(
    NAMESPACE,           // e.g., window_function
    OPERATOR_NAME,       // e.g., hann_window
    SIZE_PARAMS,         // e.g., (int64_t n) or (at::IntArrayRef shape)
    EXTRA_PARAMS,        // e.g., (bool periodic) or ()
    OUTPUT_SHAPE_EXPR    // e.g., {n} or shape
)
```

**Generated Code Pattern:**
```cpp
namespace torchscience::cpu::NAMESPACE {

inline at::Tensor OPERATOR_NAME(
    SIZE_PARAMS,
    EXTRA_PARAMS,
    const c10::optional<at::ScalarType>& dtype,
    const c10::optional<at::Layout>& layout,
    const c10::optional<at::Device>& device,
    bool requires_grad
) {
    // Validation (size >= 0, etc.)
    TORCH_CHECK(...);

    // Build options with defaults
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
        .layout(layout.value_or(at::kStrided))
        .device(device.value_or(at::kCPU))
        .requires_grad(false);

    // Handle empty case
    if (/* size == 0 */) {
        auto result = at::empty(OUTPUT_SHAPE_EXPR, options);
        if (requires_grad) result = result.requires_grad_(true);
        return result;
    }

    // Allocate output
    at::Tensor output = at::empty(OUTPUT_SHAPE_EXPR, options);

    // Dispatch to typed kernel
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        output.scalar_type(),
        #OPERATOR_NAME,
        [&]() {
            impl::NAMESPACE::OPERATOR_NAME##_kernel<scalar_t>(
                output.data_ptr<scalar_t>(),
                SIZE_PARAMS,
                EXTRA_PARAMS
            );
        }
    );

    if (requires_grad) {
        output = output.requires_grad_(true);
    }

    return output;
}

}  // namespace torchscience::cpu::NAMESPACE

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(#OPERATOR_NAME, &torchscience::cpu::NAMESPACE::OPERATOR_NAME);
}
```

**Acceptance Criteria:**
- [ ] Macro compiles without errors
- [ ] Handles empty size case (n=0)
- [ ] Validates size >= 0
- [ ] Builds TensorOptions with correct defaults
- [ ] Dispatches to typed kernel
- [ ] Handles requires_grad correctly
- [ ] Registers with CPU dispatch key

---

### Task 2: Create CUDA creation macro

**File:** `src/torchscience/csrc/cuda/creation_macros.h`

**Pattern:** Same as CPU but with CUDA kernel launch:

```cpp
AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    #OPERATOR_NAME,
    [&]() {
        const int threads = 256;
        const int blocks = (numel + threads - 1) / threads;
        impl::NAMESPACE::OPERATOR_NAME##_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            SIZE_PARAMS,
            EXTRA_PARAMS,
            numel
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
);
```

**Acceptance Criteria:**
- [ ] Correct CUDA kernel launch configuration
- [ ] Handles all supported dtypes
- [ ] Registers with CUDA dispatch key

---

### Task 3: Create Meta creation macro

**File:** `src/torchscience/csrc/meta/creation_macros.h`

**Pattern:** Shape inference only, no computation:

```cpp
inline at::Tensor OPERATOR_NAME(
    SIZE_PARAMS,
    EXTRA_PARAMS,
    const c10::optional<at::ScalarType>& dtype,
    const c10::optional<at::Layout>& layout,
    const c10::optional<at::Device>& device,
    bool requires_grad
) {
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
        .layout(layout.value_or(at::kStrided))
        .device(at::kMeta)
        .requires_grad(requires_grad);

    return at::empty(OUTPUT_SHAPE_EXPR, options);
}
```

**Acceptance Criteria:**
- [ ] Returns meta tensor with correct shape
- [ ] No computation performed
- [ ] Registers with Meta dispatch key

---

### Task 4: Create Autocast creation macro

**File:** `src/torchscience/csrc/autocast/creation_macros.h`

**Pattern:** Cast to autocast dtype:

```cpp
inline at::Tensor OPERATOR_NAME(
    SIZE_PARAMS,
    EXTRA_PARAMS,
    const c10::optional<at::ScalarType>& dtype,
    const c10::optional<at::Layout>& layout,
    const c10::optional<at::Device>& device,
    bool requires_grad
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = dtype.has_value()
        ? dtype
        : c10::optional<at::ScalarType>(at::autocast::get_autocast_dtype(at::kCUDA));

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")
        .typed<at::Tensor(SIZE_PARAMS, EXTRA_PARAMS, ...)>()
        .call(SIZE_ARGS, EXTRA_ARGS, target_dtype, layout, device, requires_grad);
}
```

**Acceptance Criteria:**
- [ ] Correctly applies autocast dtype
- [ ] Preserves explicit dtype if provided
- [ ] Registers with AutocastCUDA dispatch key

---

### Task 5: Create Sparse COO CPU creation macro

**File:** `src/torchscience/csrc/sparse/coo/cpu/creation_macros.h`

**Pattern:** Create sparse COO tensor:

```cpp
inline at::Tensor OPERATOR_NAME(
    SIZE_PARAMS,
    EXTRA_PARAMS,
    const c10::optional<at::ScalarType>& dtype,
    const c10::optional<at::Layout>& layout,  // ignored, always sparse_coo
    const c10::optional<at::Device>& device,
    bool requires_grad
) {
    auto value_dtype = dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype()));
    auto dev = device.value_or(at::kCPU);

    // Compute indices and values
    at::Tensor indices, values;
    AT_DISPATCH_FLOATING_TYPES(value_dtype, #OPERATOR_NAME, [&]() {
        std::tie(indices, values) = impl::NAMESPACE::OPERATOR_NAME##_sparse<scalar_t>(
            SIZE_PARAMS, EXTRA_PARAMS
        );
    });

    auto result = at::_sparse_coo_tensor_unsafe(
        indices.to(dev),
        values.to(dev),
        OUTPUT_SHAPE_EXPR,
        at::TensorOptions().dtype(value_dtype).device(dev)
    );

    if (requires_grad) {
        result = result.requires_grad_(true);
    }

    return result;
}
```

**Acceptance Criteria:**
- [ ] Creates valid sparse COO tensor
- [ ] Correctly computes indices and values
- [ ] Registers with SparseCPU dispatch key

---

### Task 6: Create Sparse COO CUDA creation macro

**File:** `src/torchscience/csrc/sparse/coo/cuda/creation_macros.h`

**Pattern:** Same as CPU but with CUDA device and kernels.

**Acceptance Criteria:**
- [ ] Creates sparse COO tensor on CUDA
- [ ] Registers with SparseCUDA dispatch key

---

### Task 7: Create Sparse CSR CPU creation macro

**File:** `src/torchscience/csrc/sparse/csr/cpu/creation_macros.h`

**Pattern:** Create sparse CSR tensor:

```cpp
auto result = at::_sparse_csr_tensor_unsafe(
    crow_indices.to(dev),
    col_indices.to(dev),
    values.to(dev),
    OUTPUT_SHAPE_EXPR,
    at::TensorOptions().dtype(value_dtype).device(dev).layout(at::kSparseCsr)
);
```

**Acceptance Criteria:**
- [ ] Creates valid sparse CSR tensor
- [ ] Correctly computes crow_indices, col_indices, values
- [ ] Registers with SparseCsrCPU dispatch key

---

### Task 8: Create Sparse CSR CUDA creation macro

**File:** `src/torchscience/csrc/sparse/csr/cuda/creation_macros.h`

**Pattern:** Same as CPU but with CUDA device and kernels.

**Acceptance Criteria:**
- [ ] Creates sparse CSR tensor on CUDA
- [ ] Registers with SparseCsrCUDA dispatch key

---

### Task 9: Create Quantized CPU creation macro

**File:** `src/torchscience/csrc/quantized/cpu/creation_macros.h`

**Macro Signature (extended):**
```cpp
#define QUANTIZED_CPU_CREATION_OPERATOR(
    NAMESPACE,
    OPERATOR_NAME,
    SIZE_PARAMS,
    EXTRA_PARAMS,
    OUTPUT_SHAPE_EXPR
)
```

**Pattern:** Create quantized tensor with scale/zero_point:

```cpp
inline at::Tensor OPERATOR_NAME(
    SIZE_PARAMS,
    EXTRA_PARAMS,
    double scale,
    int64_t zero_point,
    at::ScalarType qtype,  // quint8, qint8, qint32
    const c10::optional<at::Device>& device,
    bool requires_grad
) {
    TORCH_CHECK(
        qtype == at::kQUInt8 || qtype == at::kQInt8 || qtype == at::kQInt32,
        "Unsupported quantized dtype"
    );

    // Create float tensor first
    auto float_options = at::TensorOptions()
        .dtype(at::kFloat)
        .device(device.value_or(at::kCPU));

    at::Tensor float_result = at::empty(OUTPUT_SHAPE_EXPR, float_options);

    // Fill with values
    impl::NAMESPACE::OPERATOR_NAME##_kernel<float>(
        float_result.data_ptr<float>(),
        SIZE_PARAMS,
        EXTRA_PARAMS
    );

    // Quantize
    return at::quantize_per_tensor(float_result, scale, zero_point, qtype);
}
```

**Acceptance Criteria:**
- [ ] Accepts scale, zero_point, qtype parameters
- [ ] Creates valid quantized tensor
- [ ] Supports quint8, qint8, qint32
- [ ] Registers with QuantizedCPU dispatch key

---

### Task 10: Create Quantized CUDA creation macro

**File:** `src/torchscience/csrc/quantized/cuda/creation_macros.h`

**Pattern:** Same as CPU but with CUDA kernels.

**Acceptance Criteria:**
- [ ] Creates quantized tensor on CUDA
- [ ] Registers with QuantizedCUDA dispatch key

---

### Task 11: Create stochastic creation macro variants

**Files:** Add `_STOCHASTIC` variants to each macro file

**Extended signature:** Adds `generator` parameter:

```cpp
#define CPU_STOCHASTIC_CREATION_OPERATOR(
    NAMESPACE,
    OPERATOR_NAME,
    SIZE_PARAMS,
    EXTRA_PARAMS,
    OUTPUT_SHAPE_EXPR
)

// Generated function includes:
inline at::Tensor OPERATOR_NAME(
    SIZE_PARAMS,
    EXTRA_PARAMS,
    const c10::optional<at::ScalarType>& dtype,
    const c10::optional<at::Layout>& layout,
    const c10::optional<at::Device>& device,
    const c10::optional<at::Generator>& generator,  // Added
    bool requires_grad
) {
    // Use generator for random number generation
    auto gen = generator.has_value()
        ? generator.value()
        : at::detail::getDefaultCPUGenerator();
    // ...
}
```

**Acceptance Criteria:**
- [ ] Stochastic variants for CPU, CUDA, Meta, Autocast
- [ ] Correctly uses generator for RNG
- [ ] Falls back to default generator if not provided

---

### Task 12: Refactor rectangular_window to use macros

**Files:**
- `src/torchscience/csrc/cpu/signal_processing/window_function/rectangular_window.h`
- `src/torchscience/csrc/cuda/signal_processing/window_function/rectangular_window.h` (new)
- `src/torchscience/csrc/meta/signal_processing/window_function/rectangular_window.h` (new)
- Remove: `src/torchscience/csrc/composite/signal_processing/window_functions.h`

**Before:**
```cpp
// ~40 lines handwritten in composite/
inline at::Tensor rectangular_window(...) {
    // Manual implementation
}
TORCH_LIBRARY_IMPL(torchscience, CompositeImplicitAutograd, module) { ... }
```

**After:**
```cpp
// cpu/signal_processing/window_function/rectangular_window.h
#include "../../creation_macros.h"

namespace impl::window_function {
    template <typename scalar_t>
    void rectangular_window_kernel(scalar_t* output, int64_t n) {
        for (int64_t i = 0; i < n; ++i) {
            output[i] = scalar_t(1);
        }
    }
}

CPU_CREATION_OPERATOR(
    window_function,
    rectangular_window,
    (int64_t n),
    (),
    {n}
)
```

**Acceptance Criteria:**
- [ ] All existing tests pass
- [ ] Separate CPU/CUDA/Meta implementations
- [ ] Removed CompositeImplicitAutograd registration

---

### Task 13: Refactor sine_wave to use macros

**Files:**
- `src/torchscience/csrc/cpu/signal_processing/waveform/sine_wave.h`
- `src/torchscience/csrc/cuda/signal_processing/waveform/sine_wave.cu` (new)
- `src/torchscience/csrc/meta/signal_processing/waveform/sine_wave.h` (new)
- Remove: `src/torchscience/csrc/composite/signal_processing/waveform.h`

**After:**
```cpp
// cpu/signal_processing/waveform/sine_wave.h
#include "../../creation_macros.h"

namespace impl::waveform {
    template <typename scalar_t>
    void sine_wave_kernel(
        scalar_t* output,
        int64_t n,
        double frequency,
        double sample_rate,
        double amplitude,
        double phase
    ) {
        double angular_freq = 2.0 * M_PI * frequency / sample_rate;
        for (int64_t i = 0; i < n; ++i) {
            output[i] = static_cast<scalar_t>(
                amplitude * std::sin(angular_freq * i + phase)
            );
        }
    }
}

CPU_CREATION_OPERATOR(
    waveform,
    sine_wave,
    (int64_t n),
    (double frequency, double sample_rate, double amplitude, double phase),
    {n}
)
```

**Acceptance Criteria:**
- [ ] All existing tests pass
- [ ] Separate CPU/CUDA/Meta implementations
- [ ] Removed CompositeImplicitAutograd registration

---

### Task 14: Add comprehensive macro tests

**File:** `tests/torchscience/csrc/creation/test_macros.py`

**Test Cases:**
1. Shape correctness for various sizes
2. Dtype handling (all floating types)
3. Device handling (CPU, CUDA)
4. requires_grad behavior
5. Empty tensor (n=0) handling
6. Negative size error
7. Meta tensor shape inference
8. Autocast dtype casting
9. Sparse tensor structure
10. Quantized tensor properties
11. Stochastic reproducibility with generator

**Acceptance Criteria:**
- [ ] All test cases pass
- [ ] Coverage for all macro variants
- [ ] Tests for CPU, CUDA, Meta, Autocast, Sparse, Quantized

---

## File Structure After Implementation

```
src/torchscience/csrc/
├── cpu/
│   ├── creation_macros.h                    # Task 1
│   └── signal_processing/
│       ├── window_function/
│       │   └── rectangular_window.h         # Task 12
│       └── waveform/
│           └── sine_wave.h                  # Task 13
├── cuda/
│   ├── creation_macros.h                    # Task 2
│   └── signal_processing/
│       ├── window_function/
│       │   └── rectangular_window.cu        # Task 12
│       └── waveform/
│           └── sine_wave.cu                 # Task 13
├── meta/
│   └── creation_macros.h                    # Task 3
├── autocast/
│   └── creation_macros.h                    # Task 4
├── sparse/
│   ├── coo/
│   │   ├── cpu/
│   │   │   └── creation_macros.h            # Task 5
│   │   └── cuda/
│   │       └── creation_macros.h            # Task 6
│   └── csr/
│       ├── cpu/
│       │   └── creation_macros.h            # Task 7
│       └── cuda/
│           └── creation_macros.h            # Task 8
├── quantized/
│   ├── cpu/
│   │   └── creation_macros.h                # Task 9
│   └── cuda/
│       └── creation_macros.h                # Task 10
└── impl/
    ├── window_function/
    │   └── rectangular_window.h
    └── waveform/
        └── sine_wave.h
```

---

## Dependencies

```
Tasks 1-4 (CPU, CUDA, Meta, Autocast)
    │
    ├──► Task 12 (refactor rectangular_window)
    │
    └──► Task 13 (refactor sine_wave)

Tasks 5-8 (Sparse COO/CSR CPU/CUDA)
    │
    └──► (future sparse creation ops)

Tasks 9-10 (Quantized CPU/CUDA)
    │
    └──► (future quantized creation ops)

Task 11 (Stochastic variants)
    │
    └──► (future noise generators)

All tasks ──► Task 14 (tests)
```

- Tasks 1-4, 5-8, 9-10, 11 can be developed in parallel
- Tasks 12-13 require Tasks 1-4
- Task 14 requires all other tasks

---

## Estimated Complexity

| Task | Description | Files | Lines (approx) | Complexity |
|------|-------------|-------|----------------|------------|
| 1 | CPU creation macro | 1 | 150 | High |
| 2 | CUDA creation macro | 1 | 150 | High |
| 3 | Meta creation macro | 1 | 50 | Low |
| 4 | Autocast creation macro | 1 | 80 | Medium |
| 5 | Sparse COO CPU macro | 1 | 120 | Medium |
| 6 | Sparse COO CUDA macro | 1 | 120 | Medium |
| 7 | Sparse CSR CPU macro | 1 | 120 | Medium |
| 8 | Sparse CSR CUDA macro | 1 | 120 | Medium |
| 9 | Quantized CPU macro | 1 | 100 | Medium |
| 10 | Quantized CUDA macro | 1 | 100 | Medium |
| 11 | Stochastic variants | 4 | 200 | Medium |
| 12 | Refactor rectangular_window | 3 | -20 (net) | Low |
| 13 | Refactor sine_wave | 3 | -20 (net) | Low |
| 14 | Tests | 1 | 300 | Medium |

---

## Success Criteria

1. All existing tests continue to pass
2. New creation operators can be added with <30 lines of code
3. Macro-generated code matches hand-written behavior
4. Full backend coverage: CPU, CUDA, Meta, Autocast, Sparse (COO/CSR), Quantized
5. Stochastic operators support reproducible generation via `generator`
