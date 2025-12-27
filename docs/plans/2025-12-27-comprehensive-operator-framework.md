# Comprehensive Operator Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the X-macro infrastructure beyond pointwise operators to cover reductions, transforms, distance functions, graphics, and factory operators.

**Architecture:** Incremental migration from manual schema registration to X-macro-based registration, one operator category at a time. Each category gets its own `.def` file and registration templates in `core/`. CPU kernels stay in `cpu/`, but share common infrastructure.

**Tech Stack:** C++20, PyTorch TorchLib, X-macros, TORCH_LIBRARY registration

---

## Current State (as of 2025-12-27)

| Category | Example Operators | Registration Style | X-Macro? |
|----------|-------------------|-------------------|----------|
| Pointwise | gamma, chebyshev_polynomial_t | X-macro via special_functions.def | YES |
| Reduction | kurtosis | Manual in torchscience.cpp | NO |
| Transform | hilbert_transform | Manual in torchscience.cpp | NO |
| Distance | minkowski_distance | Manual in torchscience.cpp | NO |
| Graphics | cook_torrance | Manual in torchscience.cpp | NO |
| Factory | sine_wave, rectangular_window | Manual in torchscience.cpp | NO |
| Histogram | histogram, histogram_edges | Manual in torchscience.cpp | NO |

**Files that currently contain manual registrations (to be migrated):**
- `src/torchscience/csrc/torchscience.cpp:94-143` - all manual schema definitions

---

## Wave 1: Reduction X-Macro (kurtosis)

This wave creates the X-macro infrastructure for reduction operators using kurtosis as the prototype.

### Task 1: Create reductions.def X-macro registry

**Files:**
- Create: `src/torchscience/csrc/operators/reductions.def`

**Step 1: Create the reductions.def file**

```cpp
// src/torchscience/csrc/operators/reductions.def
// ============================================================================
// Reduction Operators Registry
// ============================================================================
//
// Format: X(name, extra_args_schema, extra_args_count, impl_type)
//   name            - operator name (e.g., kurtosis)
//   extra_args      - extra non-tensor arguments as schema string
//   impl_type       - fully qualified traits struct type
// ============================================================================

#ifndef TORCHSCIENCE_REDUCTIONS
#define TORCHSCIENCE_REDUCTIONS(X) \
    X(kurtosis, "bool fisher, bool bias", 2, torchscience::impl::statistics::descriptive::KurtosisImpl)
#endif
```

**Step 2: Verify the file exists**

Run: `cat src/torchscience/csrc/operators/reductions.def`
Expected: File contents visible

**Step 3: Commit**

```bash
git add src/torchscience/csrc/operators/reductions.def
git commit -m "feat(operators): add reductions.def X-macro registry"
```

---

### Task 2: Create reduction schema generator

**Files:**
- Create: `src/torchscience/csrc/core/reduction_schema.h`

**Step 1: Create the reduction schema generator**

```cpp
// src/torchscience/csrc/core/reduction_schema.h
#pragma once

#include <string>
#include <sstream>
#include <torch/library.h>

namespace torchscience::core {

// Schema generator for reduction operators with extra args
struct ReductionSchema {
    // Forward: op(Tensor input, int[]? dim, bool keepdim, <extra_args>) -> Tensor
    static std::string forward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "(Tensor input, int[]? dim, bool keepdim";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ") -> Tensor";
        return ss.str();
    }

    // Backward: op_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim, <extra_args>) -> Tensor
    static std::string backward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ") -> Tensor";
        return ss.str();
    }

    // Backward-backward: op_backward_backward(...) -> (Tensor, Tensor)
    static std::string backward_backward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "_backward_backward(Tensor grad_grad_input, Tensor grad_output, "
           << "Tensor input, int[]? dim, bool keepdim";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ") -> (Tensor, Tensor)";
        return ss.str();
    }
};

// Helper to register all schemas for a reduction operator
inline void register_reduction_schema(
    torch::Library& m,
    const char* name,
    const char* extra_args
) {
    m.def(ReductionSchema::forward(name, extra_args));
    m.def(ReductionSchema::backward(name, extra_args));
    m.def(ReductionSchema::backward_backward(name, extra_args));
}

}  // namespace torchscience::core

#define DEFINE_REDUCTION_SCHEMA(m, name, extra_args, extra_count, impl) \
    ::torchscience::core::register_reduction_schema(m, #name, extra_args)
```

**Step 2: Verify the file compiles by checking syntax**

Run: `head -20 src/torchscience/csrc/core/reduction_schema.h`
Expected: File header visible

**Step 3: Commit**

```bash
git add src/torchscience/csrc/core/reduction_schema.h
git commit -m "feat(core): add reduction schema generator"
```

---

### Task 3: Create kurtosis traits struct

**Files:**
- Create: `src/torchscience/csrc/impl/statistics/descriptive/kurtosis_traits.h`

**Step 1: Create the kurtosis traits struct**

```cpp
// src/torchscience/csrc/impl/statistics/descriptive/kurtosis_traits.h
#pragma once

#include <ATen/ATen.h>
#include <c10/core/Dispatcher.h>
#include <tuple>

namespace torchscience::impl::statistics::descriptive {

struct KurtosisImpl {
    static constexpr const char* name = "kurtosis";

    // Forward dispatch
    static at::Tensor dispatch_forward(
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        bool fisher,
        bool bias
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kurtosis", "")
            .typed<at::Tensor(
                const at::Tensor&,
                at::OptionalIntArrayRef,
                bool,
                bool,
                bool
            )>()
            .call(input, dim, keepdim, fisher, bias);
    }

    // Backward dispatch
    static at::Tensor dispatch_backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        bool fisher,
        bool bias
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kurtosis_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                at::OptionalIntArrayRef,
                bool,
                bool,
                bool
            )>()
            .call(grad_output, input, dim, keepdim, fisher, bias);
    }

    // Backward-backward dispatch
    static std::tuple<at::Tensor, at::Tensor> dispatch_backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        bool fisher,
        bool bias
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kurtosis_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                at::OptionalIntArrayRef,
                bool,
                bool,
                bool
            )>()
            .call(grad_grad_input, grad_output, input, dim, keepdim, fisher, bias);
    }
};

}  // namespace torchscience::impl::statistics::descriptive
```

**Step 2: Verify file exists**

Run: `wc -l src/torchscience/csrc/impl/statistics/descriptive/kurtosis_traits.h`
Expected: ~80 lines

**Step 3: Commit**

```bash
git add src/torchscience/csrc/impl/statistics/descriptive/kurtosis_traits.h
git commit -m "feat(impl): add kurtosis traits struct for X-macro"
```

---

### Task 4: Update torchscience.cpp to use X-macro for kurtosis schema

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp:3-4` (add includes)
- Modify: `src/torchscience/csrc/torchscience.cpp:124-128` (replace manual schema)

**Step 1: Add includes for reduction schema and reductions.def**

At line 4, after `#include "operators/special_functions.def"`, add:

```cpp
#include "core/reduction_schema.h"
#include "operators/reductions.def"
```

**Step 2: Replace manual kurtosis schema with X-macro**

Replace lines 124-128 (the kurtosis schema definitions):

```cpp
  // `torchscience.statistics.descriptive`
  module.def("kurtosis(Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> Tensor");
  module.def("kurtosis_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> Tensor");
  module.def("kurtosis_backward_backward(Tensor grad_grad_input, Tensor grad_output, Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> (Tensor, Tensor)");
```

With:

```cpp
  // `torchscience.statistics.descriptive` - auto-generated from X-macro
  #define DEFINE_REDUCTION(name, extra_args, extra_count, impl) \
      DEFINE_REDUCTION_SCHEMA(module, name, extra_args, extra_count, impl);
  TORCHSCIENCE_REDUCTIONS(DEFINE_REDUCTION)
  #undef DEFINE_REDUCTION
```

**Step 3: Build to verify no compile errors**

Run: `uv run python -c "import torchscience; print('Import OK')"`
Expected: "Import OK"

**Step 4: Run kurtosis tests to verify no regression**

Run: `uv run pytest tests/torchscience/statistics/descriptive/test__kurtosis.py -v -x`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "refactor: use X-macro for kurtosis schema registration"
```

---

### Task 5: Add integration test for reduction X-macro

**Files:**
- Create: `tests/torchscience/test__reduction_xmacro.py`

**Step 1: Create integration test**

```python
# tests/torchscience/test__reduction_xmacro.py
"""Integration tests for reduction X-macro infrastructure."""

import pytest
import torch
import torchscience


class TestReductionXMacro:
    """Test that reduction operators registered via X-macro work correctly."""

    def test_kurtosis_forward(self):
        """Test kurtosis forward pass."""
        x = torch.randn(100, dtype=torch.float64)
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert result.shape == ()
        assert result.dtype == torch.float64

    def test_kurtosis_backward(self):
        """Test kurtosis backward pass."""
        x = torch.randn(100, dtype=torch.float64, requires_grad=True)
        result = torchscience.statistics.descriptive.kurtosis(x)
        result.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_kurtosis_dim_reduction(self):
        """Test kurtosis with dimension specification."""
        x = torch.randn(10, 20, dtype=torch.float64)
        result = torchscience.statistics.descriptive.kurtosis(x, dim=1)
        assert result.shape == (10,)

    def test_kurtosis_keepdim(self):
        """Test kurtosis with keepdim=True."""
        x = torch.randn(10, 20, dtype=torch.float64)
        result = torchscience.statistics.descriptive.kurtosis(x, dim=1, keepdim=True)
        assert result.shape == (10, 1)

    def test_kurtosis_extra_args(self):
        """Test kurtosis with fisher and bias arguments."""
        x = torch.randn(100, dtype=torch.float64)

        # Fisher kurtosis (excess kurtosis)
        fisher_result = torchscience.statistics.descriptive.kurtosis(x, fisher=True)

        # Pearson kurtosis
        pearson_result = torchscience.statistics.descriptive.kurtosis(x, fisher=False)

        # Fisher should be Pearson - 3
        assert torch.allclose(fisher_result, pearson_result - 3, atol=1e-6)
```

**Step 2: Run the test**

Run: `uv run pytest tests/torchscience/test__reduction_xmacro.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/torchscience/test__reduction_xmacro.py
git commit -m "test: add integration tests for reduction X-macro"
```

---

## Wave 2: Transform X-Macro (hilbert_transform)

This wave creates X-macro infrastructure for fixed-dimension transform operators.

### Task 6: Create transforms.def X-macro registry

**Files:**
- Create: `src/torchscience/csrc/operators/transforms.def`

**Step 1: Create the transforms.def file**

```cpp
// src/torchscience/csrc/operators/transforms.def
// ============================================================================
// Transform Operators Registry
// ============================================================================
//
// Format: X(name, extra_args_schema, impl_type)
//   name         - operator name
//   extra_args   - extra parameters beyond (Tensor input, int n, int dim)
//   impl_type    - fully qualified traits struct type
// ============================================================================

#ifndef TORCHSCIENCE_TRANSFORMS
#define TORCHSCIENCE_TRANSFORMS(X) \
    X(hilbert_transform, "int padding_mode=0, float padding_value=0.0, Tensor? window=None", \
      torchscience::impl::integral_transform::HilbertTransformImpl) \
    X(inverse_hilbert_transform, "int padding_mode=0, float padding_value=0.0, Tensor? window=None", \
      torchscience::impl::integral_transform::InverseHilbertTransformImpl)
#endif
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/operators/transforms.def
git commit -m "feat(operators): add transforms.def X-macro registry"
```

---

### Task 7: Create transform schema generator

**Files:**
- Create: `src/torchscience/csrc/core/transform_schema.h`

**Step 1: Create the transform schema generator**

```cpp
// src/torchscience/csrc/core/transform_schema.h
#pragma once

#include <string>
#include <sstream>
#include <torch/library.h>

namespace torchscience::core {

// Schema generator for fixed-dimension transform operators
struct TransformSchema {
    // Forward: op(Tensor input, int n=-1, int dim=-1, <extra_args>) -> Tensor
    static std::string forward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "(Tensor input, int n=-1, int dim=-1";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ") -> Tensor";
        return ss.str();
    }

    // Backward: op_backward(Tensor grad_output, Tensor input, int n, int dim, <extra_args>) -> Tensor
    static std::string backward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor grad_output, Tensor input, int n, int dim";
        if (extra_args && extra_args[0]) {
            // Remove defaults for backward
            std::string args = extra_args;
            // Simple removal of defaults (not robust, but works for our patterns)
            size_t pos;
            while ((pos = args.find("=-1")) != std::string::npos ||
                   (pos = args.find("=0")) != std::string::npos ||
                   (pos = args.find("=0.0")) != std::string::npos ||
                   (pos = args.find("=None")) != std::string::npos) {
                size_t end = pos;
                while (end < args.size() && args[end] != ',' && args[end] != ')') {
                    ++end;
                }
                args.erase(pos, end - pos);
            }
            ss << ", " << args;
        }
        ss << ") -> Tensor";
        return ss.str();
    }

    // Backward-backward
    static std::string backward_backward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "_backward_backward(Tensor grad_grad_input, Tensor grad_output, "
           << "Tensor input, int n, int dim";
        if (extra_args && extra_args[0]) {
            std::string args = extra_args;
            size_t pos;
            while ((pos = args.find("=-1")) != std::string::npos ||
                   (pos = args.find("=0")) != std::string::npos ||
                   (pos = args.find("=0.0")) != std::string::npos ||
                   (pos = args.find("=None")) != std::string::npos) {
                size_t end = pos;
                while (end < args.size() && args[end] != ',' && args[end] != ')') {
                    ++end;
                }
                args.erase(pos, end - pos);
            }
            ss << ", " << args;
        }
        ss << ") -> (Tensor, Tensor)";
        return ss.str();
    }
};

inline void register_transform_schema(
    torch::Library& m,
    const char* name,
    const char* extra_args
) {
    m.def(TransformSchema::forward(name, extra_args));
    m.def(TransformSchema::backward(name, extra_args));
    m.def(TransformSchema::backward_backward(name, extra_args));
}

}  // namespace torchscience::core

#define DEFINE_TRANSFORM_SCHEMA(m, name, extra_args, impl) \
    ::torchscience::core::register_transform_schema(m, #name, extra_args)
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/core/transform_schema.h
git commit -m "feat(core): add transform schema generator"
```

---

### Task 8: Update torchscience.cpp to use X-macro for transforms

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp` (add includes, replace manual schemas)

**Step 1: Add includes after reductions.def include**

```cpp
#include "core/transform_schema.h"
#include "operators/transforms.def"
```

**Step 2: Replace manual hilbert_transform schemas (lines 133-142)**

Replace:

```cpp
  // `torchscience.integral_transform`
  // n=-1 means use input size along dim (no padding/truncation)
  // padding_mode: 0=constant, 1=reflect, 2=replicate, 3=circular
  module.def("hilbert_transform(Tensor input, int n=-1, int dim=-1, int padding_mode=0, float padding_value=0.0, Tensor? window=None) -> Tensor");
  module.def("hilbert_transform_backward(Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("hilbert_transform_backward_backward(Tensor grad_grad_input, Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");

  module.def("inverse_hilbert_transform(Tensor input, int n=-1, int dim=-1, int padding_mode=0, float padding_value=0.0, Tensor? window=None) -> Tensor");
  module.def("inverse_hilbert_transform_backward(Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("inverse_hilbert_transform_backward_backward(Tensor grad_grad_input, Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");
```

With:

```cpp
  // `torchscience.integral_transform` - auto-generated from X-macro
  #define DEFINE_TRANSFORM(name, extra_args, impl) \
      DEFINE_TRANSFORM_SCHEMA(module, name, extra_args, impl);
  TORCHSCIENCE_TRANSFORMS(DEFINE_TRANSFORM)
  #undef DEFINE_TRANSFORM
```

**Step 3: Build and test**

Run: `uv run python -c "import torchscience; print('Import OK')"`
Expected: "Import OK"

Run: `uv run pytest tests/torchscience/integral_transform/ -v -x --tb=short`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "refactor: use X-macro for transform schema registration"
```

---

## Wave 3: Distance X-Macro (minkowski_distance)

### Task 9: Create distance.def and pairwise schema

**Files:**
- Create: `src/torchscience/csrc/operators/distance.def`
- Create: `src/torchscience/csrc/core/pairwise_schema.h`

**Step 1: Create distance.def**

```cpp
// src/torchscience/csrc/operators/distance.def
#ifndef TORCHSCIENCE_DISTANCE
#define TORCHSCIENCE_DISTANCE(X) \
    X(minkowski_distance, "float p, Tensor? weight", \
      torchscience::impl::distance::MinkowskiDistanceImpl)
#endif
```

**Step 2: Create pairwise_schema.h**

```cpp
// src/torchscience/csrc/core/pairwise_schema.h
#pragma once

#include <string>
#include <sstream>
#include <torch/library.h>

namespace torchscience::core {

struct PairwiseDistanceSchema {
    // Forward: op(Tensor x, Tensor y, <extra_args>) -> Tensor
    static std::string forward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "(Tensor x, Tensor y";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ") -> Tensor";
        return ss.str();
    }

    // Backward: op_backward(Tensor grad_output, Tensor x, Tensor y, <extra_args>, Tensor output) -> (Tensor, Tensor, Tensor)
    static std::string backward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor grad_output, Tensor x, Tensor y";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ", Tensor dist_output) -> (Tensor, Tensor, Tensor)";
        return ss.str();
    }
};

inline void register_pairwise_distance_schema(
    torch::Library& m,
    const char* name,
    const char* extra_args
) {
    m.def(PairwiseDistanceSchema::forward(name, extra_args));
    m.def(PairwiseDistanceSchema::backward(name, extra_args));
}

}  // namespace torchscience::core

#define DEFINE_PAIRWISE_DISTANCE_SCHEMA(m, name, extra_args, impl) \
    ::torchscience::core::register_pairwise_distance_schema(m, #name, extra_args)
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/operators/distance.def src/torchscience/csrc/core/pairwise_schema.h
git commit -m "feat(operators): add distance.def and pairwise schema generator"
```

---

### Task 10: Update torchscience.cpp for distance X-macro

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp`

**Step 1: Add includes**

```cpp
#include "core/pairwise_schema.h"
#include "operators/distance.def"
```

**Step 2: Replace minkowski_distance manual schemas**

Replace lines 94-96:

```cpp
  // `torchscience.distance`
  module.def("minkowski_distance(Tensor x, Tensor y, float p, Tensor? weight) -> Tensor");
  module.def("minkowski_distance_backward(Tensor grad_output, Tensor x, Tensor y, float p, Tensor? weight, Tensor dist_output) -> (Tensor, Tensor, Tensor)");
```

With:

```cpp
  // `torchscience.distance` - auto-generated from X-macro
  #define DEFINE_DISTANCE(name, extra_args, impl) \
      DEFINE_PAIRWISE_DISTANCE_SCHEMA(module, name, extra_args, impl);
  TORCHSCIENCE_DISTANCE(DEFINE_DISTANCE)
  #undef DEFINE_DISTANCE
```

**Step 3: Build and test**

Run: `uv run pytest tests/torchscience/distance/ -v -x --tb=short`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "refactor: use X-macro for distance schema registration"
```

---

## Wave 4: Graphics X-Macro (cook_torrance)

### Task 11: Create graphics.def and batched graphics schema

**Files:**
- Create: `src/torchscience/csrc/operators/graphics.def`
- Create: `src/torchscience/csrc/core/graphics_schema.h`

**Step 1: Create graphics.def**

```cpp
// src/torchscience/csrc/operators/graphics.def
#ifndef TORCHSCIENCE_GRAPHICS
#define TORCHSCIENCE_GRAPHICS(X) \
    X(cook_torrance, "Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0", \
      torchscience::impl::graphics::shading::CookTorranceImpl)
#endif
```

**Step 2: Create graphics_schema.h**

```cpp
// src/torchscience/csrc/core/graphics_schema.h
#pragma once

#include <string>
#include <sstream>
#include <torch/library.h>

namespace torchscience::core {

struct GraphicsSchema {
    // Forward
    static std::string forward(const char* name, const char* tensor_args) {
        std::ostringstream ss;
        ss << name << "(" << tensor_args << ") -> Tensor";
        return ss.str();
    }

    // Backward - returns tuple of gradients for each input tensor
    static std::string backward(const char* name, const char* tensor_args, int num_tensors) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor grad_output, " << tensor_args << ") -> (";
        for (int i = 0; i < num_tensors; ++i) {
            if (i > 0) ss << ", ";
            ss << "Tensor";
        }
        ss << ")";
        return ss.str();
    }

    // Backward-backward
    static std::string backward_backward(const char* name, const char* tensor_args, int num_tensors) {
        std::ostringstream ss;
        ss << name << "_backward_backward(";
        // grad-grad inputs
        for (int i = 0; i < num_tensors; ++i) {
            ss << "Tensor gg_" << i << ", ";
        }
        ss << "Tensor grad_output, " << tensor_args << ") -> (";
        // Returns: grad-grad output + new grads for each input
        for (int i = 0; i <= num_tensors; ++i) {
            if (i > 0) ss << ", ";
            ss << "Tensor";
        }
        ss << ")";
        return ss.str();
    }
};

}  // namespace torchscience::core
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/operators/graphics.def src/torchscience/csrc/core/graphics_schema.h
git commit -m "feat(operators): add graphics.def and graphics schema generator"
```

---

### Task 12: Update torchscience.cpp for graphics X-macro

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp`

**Step 1: Add includes**

```cpp
#include "core/graphics_schema.h"
#include "operators/graphics.def"
```

**Step 2: Replace cook_torrance manual schemas**

Replace:

```cpp
  // `torchscience.graphics.shading`
  module.def("cook_torrance(Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> Tensor");
  module.def("cook_torrance_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  module.def("cook_torrance_backward_backward(Tensor gg_normal, Tensor gg_view, Tensor gg_light, Tensor gg_roughness, Tensor gg_f0, Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
```

With X-macro expansion using the graphics schema.

**Step 3: Build and test**

Run: `uv run pytest tests/torchscience/graphics/ -v -x --tb=short`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "refactor: use X-macro for graphics schema registration"
```

---

## Wave 5: Update Documentation

### Task 13: Expand adding-operators.md documentation

**Files:**
- Modify: `docs/contributing/adding-operators.md`

**Step 1: Add sections for each operator category**

Add comprehensive examples for:
- Reduction operators (kurtosis pattern)
- Transform operators (hilbert_transform pattern)
- Distance operators (minkowski_distance pattern)
- Graphics operators (cook_torrance pattern)

**Step 2: Commit**

```bash
git add docs/contributing/adding-operators.md
git commit -m "docs: expand adding-operators guide for all operator categories"
```

---

## Success Criteria

After completing all tasks:

1. **Schema generation is automated** - All operators use X-macro for schema definitions
2. **Manual lines removed** - torchscience.cpp has no manual `module.def()` calls except X-macro expansions
3. **Tests pass** - All existing tests continue to pass
4. **Documentation complete** - Guide covers all operator categories

---

## Verification Commands

```bash
# Full test suite
uv run pytest tests/ -v --tb=short

# Check no manual module.def() calls remain (except X-macro)
grep -n "module.def(" src/torchscience/csrc/torchscience.cpp | grep -v "#define"

# Build from scratch
uv run pip install -e . --no-build-isolation --force-reinstall
```
