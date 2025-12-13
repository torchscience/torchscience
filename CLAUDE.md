# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

torchscience is a PyTorch extension library providing special mathematical functions (gamma, beta, Bessel functions, elliptic integrals, etc.) as native PyTorch operators with autograd support.

## Build Commands

```bash
# Install dependencies and build
uv sync

# Run Python tests
uv run pytest

# Run specific test
uv run pytest tests/special_functions/test__gamma.py

# Build C++ tests (from build directory)
cmake .. && make torchscience_tests
./torchscience_tests
```

## Architecture

### C++ Layer (`src/torchscience/csrc/`)

Each special function has implementations across multiple dispatch backends:

- **`impl/special_functions/`** - Core mathematical implementations using Boost.Math. Contains the actual computation logic with forward and backward functions.
- **`cpu/special_functions/`** - CPU kernel registration using `TORCHSCIENCE_*_CPU_KERNEL` macros
- **`cuda/special_functions/`** - CUDA kernel registration (when available)
- **`autocast/special_functions/`** - Mixed-precision autocast support
- **`autograd/special_functions/`** - Automatic differentiation registration
- **`meta/special_functions/`** - Shape inference for lazy tensors
- **`quantized/cpu/special_functions/`** - Quantized tensor support
- **`sparse/coo/cpu/special_functions/`** - Sparse COO tensor support
- **`sparse/csr/cpu/special_functions/`** - Sparse CSR tensor support

**`special_functions.cpp`** - Registers all operators with PyTorch's dispatcher via `TORCH_LIBRARY_FRAGMENT`.

### Python Layer (`src/torchscience/special_functions/`)

Each function (e.g., `_gamma.py`) wraps `torch.ops.torchscience._<function_name>` with documentation and an optional `out` parameter.

### Namespace Convention

All C++ headers in `special_functions/` directories must be wrapped in namespaces matching their path:
```cpp
namespace torchscience::<backend>::special_functions {
// content
} // namespace torchscience::<backend>::special_functions
```

Include paths use angle brackets: `#include <torchscience/csrc/...>`

### Adding a New Special Function

1. Create `impl/special_functions/<name>.h` with core implementation using Boost.Math
2. Create corresponding files in `cpu/`, `autocast/`, `autograd/`, `meta/`, `quantized/cpu/`, `sparse/coo/cpu/`, `sparse/csr/cpu/` using the appropriate macros
3. Add operator definitions to `special_functions.cpp`
4. Create Python wrapper in `special_functions/_<name>.py`
5. Export from `special_functions/__init__.py`
6. Add tests in `tests/special_functions/test__<name>.py`

## Testing

Tests use hypothesis for property-based testing. The `conftest.py` provides base test classes (`UnaryOperatorTestCase`, `BinaryOperatorTestCase`) with common test patterns for mathematical properties like symmetry, known values, and gradient checking.