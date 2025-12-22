# Implementation Plan: Reduction Operator Macros

## Overview

Create a macro system for reduction operators that parallels the existing elementwise operator macros (`CPU_UNARY_OPERATOR`, etc.). These macros will generate boilerplate for operators that reduce tensor dimensions (like `mean`, `sum`, `std`, `kurtosis`).

## Scope

**In Scope:**
- Macros for CPU, CUDA, Meta, Autograd, Autocast backends
- Macros for Quantized CPU backend
- Macros for Sparse COO (CPU, CUDA) backends
- Macros for Sparse CSR (CPU, CUDA) backends
- Support for `dim` (optional int array), `keepdim` (bool) parameters
- Support for additional operator-specific scalar parameters
- Forward, backward, and backward_backward generation
- Dispatcher registration

**Out of Scope:**
- Python wrapper generation (remains manual)
- Schema registration (remains in `torchscience.cpp`)

## Reference Implementation

The `kurtosis` operator serves as the reference:
- **CPU:** `src/torchscience/csrc/cpu/statistics/descriptive/kurtosis.h`
- **Autograd:** `src/torchscience/csrc/autograd/statistics/descriptive/kurtosis.h`
- **Meta:** `src/torchscience/csrc/meta/statistics/descriptive/kurtosis.h`
- **Impl:** `src/torchscience/csrc/impl/statistics/descriptive/kurtosis.h`

---

## Tasks

### Task 1: Create CPU reduction macro

**File:** `src/torchscience/csrc/cpu/reduction_macros.h`

**Purpose:** Generate CPU implementations for reduction operators

**Macro Signature:**
```cpp
#define CPU_REDUCTION_OPERATOR(
    NAMESPACE,           // e.g., "descriptive"
    OPERATOR_NAME,       // e.g., "mean"
    EXTRA_PARAMS,        // e.g., (bool bias, bool fisher) or ()
    EXTRA_PARAM_TYPES,   // e.g., (bool, bool) or ()
    EXTRA_PARAM_NAMES    // e.g., (bias, fisher) or ()
)
```

**Generated Functions:**
1. `OPERATOR_NAME(input, dim, keepdim, EXTRA_PARAMS...)` - forward
2. `OPERATOR_NAME_backward(grad_output, input, dim, keepdim, EXTRA_PARAMS...)` - backward
3. `OPERATOR_NAME_backward_backward(...)` - double backward

**Pattern:**
```cpp
inline at::Tensor OPERATOR_NAME(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    EXTRA_PARAM_TYPES... EXTRA_PARAM_NAMES
) {
    auto [batch_size, reduce_size] = reduction::compute_reduce_info(input, dim);
    auto output_shape = reduction::compute_output_shape(input, dim, keepdim);
    auto permutation = reduction::compute_reduction_permutation(input, dim);

    at::Tensor permuted = input.permute(permutation).contiguous();
    at::Tensor reshaped = permuted.view({batch_size, reduce_size});

    at::Tensor output = at::empty(output_shape, input.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        input.scalar_type(),
        #OPERATOR_NAME,
        [&]() {
            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    output_ptr[b] = impl::NAMESPACE::OPERATOR_NAME<scalar_t>(
                        data_ptr + b * reduce_size,
                        reduce_size,
                        EXTRA_PARAM_NAMES...
                    );
                }
            });
        }
    );

    return output;
}
```

**Acceptance Criteria:**
- [ ] Macro compiles without errors
- [ ] Generates forward, backward, backward_backward
- [ ] Registers with `TORCH_LIBRARY_IMPL(torchscience, CPU, module)`
- [ ] Handles empty extra params case
- [ ] Parallel execution over batch dimension

---

### Task 2: Create Meta reduction macro

**File:** `src/torchscience/csrc/meta/reduction_macros.h`

**Purpose:** Generate Meta implementations for shape inference

**Macro Signature:** Same as CPU

**Generated Functions:**
```cpp
inline at::Tensor OPERATOR_NAME(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    EXTRA_PARAM_TYPES... EXTRA_PARAM_NAMES  // unused, marked [[maybe_unused]]
) {
    auto output_shape = reduction::compute_output_shape(input, dim, keepdim);
    return at::empty(output_shape, input.options());
}
```

**Acceptance Criteria:**
- [ ] Correct output shape for all dim/keepdim combinations
- [ ] Registers with `TORCH_LIBRARY_IMPL(torchscience, Meta, module)`
- [ ] No computation performed (shape only)

---

### Task 3: Create Autograd reduction macro

**File:** `src/torchscience/csrc/autograd/reduction_macros.h`

**Purpose:** Generate autograd Function classes with double-backward support

**Key Challenge:** `OptionalIntArrayRef` doesn't persist - must convert to `std::vector<int64_t>`

**Generated Classes:**
1. `CLASS_NAMEBackward` - handles backward → backward_backward
2. `CLASS_NAME` - handles forward → backward

**Pattern for saving dim:**
```cpp
// In forward:
if (dim.has_value()) {
    std::vector<int64_t> dim_vec(dim->begin(), dim->end());
    context->saved_data["dim"] = dim_vec;
    context->saved_data["has_dim"] = true;
} else {
    context->saved_data["has_dim"] = false;
}

// In backward:
bool has_dim = context->saved_data["has_dim"].toBool();
std::vector<int64_t> dim_vec;
at::OptionalIntArrayRef dim_ref;
if (has_dim) {
    dim_vec = context->saved_data["dim"].toIntVector();
    dim_ref = dim_vec;
}
```

**Acceptance Criteria:**
- [ ] `dim` parameter correctly saved and restored
- [ ] Extra params correctly saved and restored
- [ ] Double-backward support via nested Function class
- [ ] Registers with `TORCH_LIBRARY_IMPL(torchscience, Autograd, module)`

---

### Task 4: Create Autocast reduction macro

**File:** `src/torchscience/csrc/autocast/reduction_macros.h`

**Purpose:** Handle automatic mixed precision

**Pattern:**
```cpp
inline at::Tensor OPERATOR_NAME(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    EXTRA_PARAM_TYPES... EXTRA_PARAM_NAMES
) {
    // Reconstruct dim_ref from persistent vector
    std::vector<int64_t> dim_vec;
    at::OptionalIntArrayRef dim_ref;
    if (dim.has_value()) {
        dim_vec = std::vector<int64_t>(dim->begin(), dim->end());
        dim_ref = dim_vec;
    }

    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    at::Tensor input_cast = at::autocast::cached_cast(
        at::autocast::get_autocast_dtype(at::kCUDA),
        input
    );

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")
        .typed<...>()
        .call(input_cast, dim_ref, keepdim, EXTRA_PARAM_NAMES...);
}
```

**Acceptance Criteria:**
- [ ] Correctly casts input to autocast dtype
- [ ] Preserves dim parameter through dispatch
- [ ] Registers with `TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module)`

---

### Task 5: Create CUDA reduction macro

**File:** `src/torchscience/csrc/cuda/reduction_macros.h`

**Purpose:** Generate CUDA kernel launchers for reduction operators

**Approach:** The macro generates the kernel launch wrapper; actual kernels are in impl files

**Pattern:**
```cpp
inline at::Tensor OPERATOR_NAME(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    EXTRA_PARAM_TYPES... EXTRA_PARAM_NAMES
) {
    auto [batch_size, reduce_size] = reduction::compute_reduce_info(input, dim);
    auto output_shape = reduction::compute_output_shape(input, dim, keepdim);

    at::Tensor output = at::empty(output_shape, input.options());

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        input.scalar_type(),
        #OPERATOR_NAME "_cuda",
        [&]() {
            impl::NAMESPACE::OPERATOR_NAME##_cuda<scalar_t>(
                input, output, dim, keepdim, EXTRA_PARAM_NAMES...
            );
        }
    );

    return output;
}
```

**Acceptance Criteria:**
- [ ] Correct kernel launch configuration
- [ ] Handles all supported dtypes
- [ ] Registers with `TORCH_LIBRARY_IMPL(torchscience, CUDA, module)`

---

### Task 6: Refactor kurtosis to use new macros

**Purpose:** Validate macros work correctly by migrating existing operator

**Files to Modify:**
- `src/torchscience/csrc/cpu/statistics/descriptive/kurtosis.h`
- `src/torchscience/csrc/meta/statistics/descriptive/kurtosis.h`
- `src/torchscience/csrc/autograd/statistics/descriptive/kurtosis.h`
- `src/torchscience/csrc/autocast/statistics/descriptive/kurtosis.h`
- `src/torchscience/csrc/cuda/statistics/descriptive/kurtosis.cu`

**Before:**
~500 lines of handwritten code per backend

**After:**
```cpp
#include "../reduction_macros.h"
#include "../../impl/statistics/descriptive/kurtosis.h"

CPU_REDUCTION_OPERATOR(
    descriptive,
    kurtosis,
    (bool fisher, bool bias),
    (bool, bool),
    (fisher, bias)
)
```

**Acceptance Criteria:**
- [ ] All existing kurtosis tests pass
- [ ] No behavior changes
- [ ] Code reduction of ~80%

---

### Task 7: Add comprehensive macro tests

**File:** `tests/torchscience/csrc/reduction/test_macros.cpp` (or Python equivalent)

**Test Cases:**
1. Shape inference with various dim/keepdim combinations
2. Gradient correctness via `torch.autograd.gradcheck`
3. Double-backward via `torch.autograd.gradgradcheck`
4. Mixed precision behavior
5. Edge cases: empty tensors, single element, all dims

**Acceptance Criteria:**
- [ ] All test cases pass
- [ ] Coverage for CPU and CUDA backends
- [ ] Coverage for all dtype combinations

---

### Task 8: Create Quantized CPU reduction macro

**File:** `src/torchscience/csrc/quantized/cpu/reduction_macros.h`

**Purpose:** Generate quantized CPU implementations via dequantize → dispatch → requantize pattern

**Pattern:**
```cpp
inline at::Tensor OPERATOR_NAME(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    EXTRA_PARAM_TYPES... EXTRA_PARAM_NAMES
) {
    TORCH_CHECK(input.is_quantized(), #OPERATOR_NAME " expects quantized tensor");

    // Persist dim across dispatch
    std::vector<int64_t> dim_vec;
    at::OptionalIntArrayRef dim_ref;
    if (dim.has_value()) {
        dim_vec = std::vector<int64_t>(dim->begin(), dim->end());
        dim_ref = dim_vec;
    }

    at::Tensor dequantized = input.dequantize();

    at::Tensor result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")
        .typed<at::Tensor(const at::Tensor&, at::OptionalIntArrayRef, bool, EXTRA_PARAM_TYPES...)>()
        .call(dequantized, dim_ref, keepdim, EXTRA_PARAM_NAMES...);

    // Note: Reduction output may have different shape, use input's quantization params
    return at::quantize_per_tensor(
        result,
        input.q_scale(),
        input.q_zero_point(),
        input.scalar_type()
    );
}
```

**Key Considerations:**
- Reduction output shape differs from input - quantization params inherited from input
- `dim` must be persisted via `std::vector` before dispatch
- Backward/backward_backward follow same dequantize → dispatch → requantize pattern

**Acceptance Criteria:**
- [ ] Correctly dequantizes input before dispatch
- [ ] Correctly requantizes output after dispatch
- [ ] Handles dim persistence across dispatch
- [ ] Registers with `TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module)`

---

### Task 9: Create Sparse COO CPU reduction macro

**File:** `src/torchscience/csrc/sparse/coo/cpu/reduction_macros.h`

**Purpose:** Generate sparse COO CPU implementations

**Key Challenge:** Reduction semantics on sparse tensors:
- Reducing over dense dimensions: operate on `_values()` tensor
- Reducing over sparse dimensions: may require coalescing or densification

**Pattern (for reductions over dense dimensions):**
```cpp
inline at::Tensor OPERATOR_NAME(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    EXTRA_PARAM_TYPES... EXTRA_PARAM_NAMES
) {
    TORCH_CHECK(input.is_sparse(), #OPERATOR_NAME " expects sparse COO tensor");

    // For sparse tensors, reduction over value dimensions operates on values
    at::Tensor values = input._values();

    // Persist dim
    std::vector<int64_t> dim_vec;
    at::OptionalIntArrayRef dim_ref;
    if (dim.has_value()) {
        dim_vec = std::vector<int64_t>(dim->begin(), dim->end());
        dim_ref = dim_vec;
    }

    // Dispatch to dense implementation on values
    at::Tensor new_values = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")
        .typed<at::Tensor(const at::Tensor&, at::OptionalIntArrayRef, bool, EXTRA_PARAM_TYPES...)>()
        .call(values, dim_ref, keepdim, EXTRA_PARAM_NAMES...);

    // Reconstruct sparse tensor with new values
    // Note: Output shape calculation depends on reduction dims
    return at::_sparse_coo_tensor_unsafe(
        input._indices(),
        new_values,
        compute_sparse_output_sizes(input, dim, keepdim),
        input.options().dtype(new_values.scalar_type())
    )._coalesced_(input.is_coalesced());
}
```

**Acceptance Criteria:**
- [ ] Correctly extracts values for reduction
- [ ] Handles dim persistence across dispatch
- [ ] Correctly reconstructs sparse tensor with reduced values
- [ ] Preserves coalesced state
- [ ] Registers with `TORCH_LIBRARY_IMPL(torchscience, SparseCPU, module)`

---

### Task 10: Create Sparse COO CUDA reduction macro

**File:** `src/torchscience/csrc/sparse/coo/cuda/reduction_macros.h`

**Purpose:** Generate sparse COO CUDA implementations

**Pattern:** Same as CPU variant but registers with different dispatch key

**Acceptance Criteria:**
- [ ] Same functionality as CPU variant
- [ ] Registers with `TORCH_LIBRARY_IMPL(torchscience, SparseCUDA, module)`

---

### Task 11: Create Sparse CSR CPU reduction macro

**File:** `src/torchscience/csrc/sparse/csr/cpu/reduction_macros.h`

**Purpose:** Generate sparse CSR CPU implementations

**Pattern:**
```cpp
inline at::Tensor OPERATOR_NAME(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    EXTRA_PARAM_TYPES... EXTRA_PARAM_NAMES
) {
    TORCH_CHECK(input.layout() == at::kSparseCsr,
        #OPERATOR_NAME " expects sparse CSR tensor");

    at::Tensor values = input.values();

    // Persist dim
    std::vector<int64_t> dim_vec;
    at::OptionalIntArrayRef dim_ref;
    if (dim.has_value()) {
        dim_vec = std::vector<int64_t>(dim->begin(), dim->end());
        dim_ref = dim_vec;
    }

    at::Tensor new_values = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::" #OPERATOR_NAME, "")
        .typed<at::Tensor(const at::Tensor&, at::OptionalIntArrayRef, bool, EXTRA_PARAM_TYPES...)>()
        .call(values, dim_ref, keepdim, EXTRA_PARAM_NAMES...);

    return at::_sparse_csr_tensor_unsafe(
        input.crow_indices(),
        input.col_indices(),
        new_values,
        compute_csr_output_sizes(input, dim, keepdim),
        input.options().layout(at::kSparseCsr).dtype(new_values.scalar_type())
    );
}
```

**Acceptance Criteria:**
- [ ] Correctly extracts values for reduction
- [ ] Handles CSR-specific index tensors (crow_indices, col_indices)
- [ ] Registers with `TORCH_LIBRARY_IMPL(torchscience, SparseCsrCPU, module)`

---

### Task 12: Create Sparse CSR CUDA reduction macro

**File:** `src/torchscience/csrc/sparse/csr/cuda/reduction_macros.h`

**Purpose:** Generate sparse CSR CUDA implementations

**Pattern:** Same as CPU variant but registers with different dispatch key

**Acceptance Criteria:**
- [ ] Same functionality as CPU variant
- [ ] Registers with `TORCH_LIBRARY_IMPL(torchscience, SparseCsrCUDA, module)`

---

### Task 13: Update kurtosis refactor to include all backends

**Purpose:** Extend Task 6 to also refactor quantized and sparse implementations

**Additional Files to Modify:**
- `src/torchscience/csrc/quantized/cpu/statistics/descriptive/kurtosis.h`
- `src/torchscience/csrc/sparse/coo/cpu/statistics/descriptive/kurtosis.h`
- `src/torchscience/csrc/sparse/coo/cuda/statistics/descriptive/kurtosis.h`
- `src/torchscience/csrc/sparse/csr/cpu/statistics/descriptive/kurtosis.h`
- `src/torchscience/csrc/sparse/csr/cuda/statistics/descriptive/kurtosis.h`

**Acceptance Criteria:**
- [ ] All backend implementations use macros
- [ ] All existing tests pass
- [ ] Consistent behavior across all backends

---

### Task 14: Add quantized and sparse macro tests

**Purpose:** Extend Task 7 with tests for quantized and sparse backends

**Additional Test Cases:**
1. Quantized tensor reduction correctness
2. Quantization parameter preservation
3. Sparse COO reduction (preserving sparsity structure)
4. Sparse CSR reduction (preserving CSR structure)
5. Edge cases: empty sparse tensors, single non-zero element

**Acceptance Criteria:**
- [ ] Tests for QuantizedCPU backend
- [ ] Tests for SparseCPU and SparseCUDA backends
- [ ] Tests for SparseCsrCPU and SparseCsrCUDA backends

---

## File Structure After Implementation

```
src/torchscience/csrc/
├── cpu/
│   └── reduction_macros.h             # CPU macro (Task 1)
├── cuda/
│   └── reduction_macros.h             # CUDA macro (Task 5)
├── meta/
│   └── reduction_macros.h             # Meta macro (Task 2)
├── autograd/
│   └── reduction_macros.h             # Autograd macro (Task 3)
├── autocast/
│   └── reduction_macros.h             # Autocast macro (Task 4)
├── quantized/
│   └── cpu/
│       └── reduction_macros.h         # Quantized CPU macro (Task 8)
└── sparse/
    ├── coo/
    │   ├── cpu/
    │   │   └── reduction_macros.h     # Sparse COO CPU macro (Task 9)
    │   └── cuda/
    │       └── reduction_macros.h     # Sparse COO CUDA macro (Task 10)
    └── csr/
        ├── cpu/
        │   └── reduction_macros.h     # Sparse CSR CPU macro (Task 11)
        └── cuda/
            └── reduction_macros.h     # Sparse CSR CUDA macro (Task 12)
```

---

## Usage Example

After implementation, adding a new reduction operator like `variance`:

**1. Define impl function** (`impl/statistics/descriptive/variance.h`):
```cpp
template <typename T>
T variance_1d(const T* data, int64_t n, bool unbiased) {
    // Implementation
}

template <typename T>
T variance_1d_backward(...) { ... }

template <typename T>
std::tuple<T, T> variance_1d_backward_backward(...) { ... }
```

**2. Register schema** (`torchscience.cpp`):
```cpp
module.def("variance(Tensor input, int[]? dim, bool keepdim, bool unbiased) -> Tensor");
module.def("variance_backward(...) -> Tensor");
module.def("variance_backward_backward(...) -> (Tensor, Tensor)");
```

**3. Use macros** (one file per backend):
```cpp
// cpu/statistics/descriptive/variance.h
#include "../../reduction_macros.h"
CPU_REDUCTION_OPERATOR(descriptive, variance, (bool unbiased), (bool), (unbiased))

// meta/statistics/descriptive/variance.h
#include "../../reduction_macros.h"
META_REDUCTION_OPERATOR(descriptive, variance, (bool unbiased), (bool), (unbiased))

// autograd/statistics/descriptive/variance.h
#include "../../reduction_macros.h"
AUTOGRAD_REDUCTION_OPERATOR(descriptive, Variance, variance, (bool unbiased), (bool), (unbiased))
```

**4. Add Python wrapper** (`statistics/descriptive/_variance.py`):
```python
def variance(input, dim=None, keepdim=False, *, unbiased=True):
    return torch.ops.torchscience.variance(input, dim, keepdim, unbiased)
```

---

## Dependencies

```
Tasks 1-5 (CPU, Meta, Autograd, Autocast, CUDA)
    │
    └──► Task 6 (refactor kurtosis core backends)
             │
             └──► Task 7 (core backend tests)

Tasks 8-12 (Quantized, Sparse COO/CSR CPU/CUDA)
    │
    └──► Task 13 (refactor kurtosis all backends)
             │
             └──► Task 14 (quantized/sparse tests)
```

- Tasks 1-5 and Tasks 8-12 can be developed in parallel
- Task 6 requires Tasks 1-5
- Task 7 requires Task 6
- Task 13 requires Tasks 6 + 8-12
- Task 14 requires Tasks 7 + 13

## Estimated Complexity

| Task | Description                    | Files | Lines (approx) | Complexity |
|------|--------------------------------|-------|----------------|------------|
| 1    | CPU macro                      | 1     | 300            | High       |
| 2    | Meta macro                     | 1     | 100            | Low        |
| 3    | Autograd macro                 | 1     | 400            | High       |
| 4    | Autocast macro                 | 1     | 150            | Medium     |
| 5    | CUDA macro                     | 1     | 200            | Medium     |
| 6    | Refactor kurtosis (core)       | 5     | -300 (net)     | Medium     |
| 7    | Core backend tests             | 1     | 200            | Medium     |
| 8    | Quantized CPU macro            | 1     | 150            | Medium     |
| 9    | Sparse COO CPU macro           | 1     | 200            | Medium     |
| 10   | Sparse COO CUDA macro          | 1     | 100            | Low        |
| 11   | Sparse CSR CPU macro           | 1     | 200            | Medium     |
| 12   | Sparse CSR CUDA macro          | 1     | 100            | Low        |
| 13   | Refactor kurtosis (all)        | 5     | -200 (net)     | Medium     |
| 14   | Quantized/sparse tests         | 1     | 300            | Medium     |

## Success Criteria

1. All existing tests continue to pass
2. New reduction operators can be added with <50 lines of code per backend
3. Macro-generated code is equivalent to hand-written code
4. No performance regression vs hand-written implementations
5. Consistent behavior across all backends (CPU, CUDA, Meta, Autograd, Autocast, Quantized, Sparse)
