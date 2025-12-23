# Migration Guide: Macro to Template-Based Creation Operators

## Overview

This guide explains how to migrate from `CPU_CREATION_OPERATOR` macros to
`CPUCreationOperator<Traits>` templates.

## Step 1: Create a Traits Class

For each operator, create a traits class in `impl/<namespace>/<operator>_traits.h`:

```cpp
struct MyOperatorTraits {
    static std::vector<int64_t> output_shape(/* params */) {
        return {/* shape */};
    }

    template<typename scalar_t>
    static void kernel(scalar_t* output, int64_t numel, /* params */) {
        // Fill output
    }
};
```

## Step 2: Register Using Template

Replace macro with template registration:

```cpp
// Before (macro)
CPU_CREATION_OPERATOR(namespace, my_op, {n}, (int64_t n), (n))

// After (template)
#include "cpu/creation_operators.h"
#include "impl/namespace/my_op_traits.h"

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    REGISTER_CPU_CREATION(module, my_op, MyOpTraits, int64_t);
}
```

## Step 3: Run Tests

Verify existing tests still pass after migration.
