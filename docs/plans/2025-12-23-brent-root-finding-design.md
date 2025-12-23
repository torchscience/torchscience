# Brent's Root-Finding Method Design

## Overview

Add `torchscience.root_finding.brent` - a batched, differentiable implementation of Brent's root-finding method for PyTorch tensors.

## API

```python
from torch import Tensor
from typing import Callable

def brent(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    *,
    xtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 100,
) -> Tensor:
    """
    Find roots of f(x) = 0 using Brent's method.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vectorized function. Takes tensor of shape (N,), returns (N,).
    a, b : Tensor
        Bracket endpoints. Shape (N,). Must satisfy f(a) * f(b) < 0.
    xtol : float, optional
        Tolerance on interval width. Default: dtype-aware (1e-6 for float32, 1e-12 for float64).
    ftol : float, optional
        Tolerance on |f(x)|. Default: dtype-aware (1e-6 for float32, 1e-12 for float64).
    maxiter : int
        Maximum iterations. Raises RuntimeError if exceeded. Default: 100.

    Returns
    -------
    Tensor
        Roots of shape (N,). Supports autograd via implicit differentiation.

    Raises
    ------
    ValueError
        If f(a) and f(b) have the same sign for any element.
    RuntimeError
        If convergence is not achieved within maxiter iterations.
    """
```

### Usage Example

```python
import torch
from torchscience.root_finding import brent

# Find x where x^2 = c for multiple values of c
c = torch.tensor([2.0, 3.0, 4.0])
f = lambda x: x**2 - c
roots = brent(f, torch.zeros(3), torch.full((3,), 10.0))
# roots ≈ [1.414, 1.732, 2.0]
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Module location | `torchscience.root_finding` | Top-level module emphasizes general-purpose utility |
| Batching | Yes, shape `(N,)` | Leverages PyTorch parallelism |
| Function interface | Single vectorized function | PyTorch-native, simple API |
| Convergence criteria | Both `xtol` AND `ftol` | Conservative, guarantees both tight interval and small residual |
| Invalid brackets | Raise `ValueError` | Fail fast with clear feedback |
| Autograd | Implicit differentiation | Memory-efficient, numerically stable |
| Return type | Root tensor only | Simple API for common case |
| Default tolerances | Dtype-aware | Respects precision limits of each dtype |
| Max iterations | 100 (raises on exceed) | Prevents infinite loops, makes failures explicit |
| Implementation | C++/CUDA backend | Consistent with torchscience patterns |

## Module Structure

### Python

```
torchscience/
├── root_finding/
│   ├── __init__.py      # exports brent
│   └── _brent.py        # Python wrapper
```

### C++/CUDA Backend

```
src/torchscience/csrc/
├── impl/root_finding/
│   └── brent.h              # Core algorithm (device-agnostic template)
├── cpu/root_finding/
│   └── brent.h              # CPU kernel dispatch
├── cuda/root_finding/
│   └── brent.cu             # CUDA kernel
├── meta/root_finding/
│   └── brent.h              # Meta tensor support (shape inference)
├── autograd/root_finding/
│   └── brent.h              # Backward pass via implicit differentiation
└── autocast/root_finding/
    └── brent.h              # Mixed precision handling
```

## Iteration Protocol

Since C++ cannot call Python functions directly, we use a stateful iteration protocol:

```python
def brent(f, a, b, *, xtol=None, ftol=None, maxiter=100):
    # Validate brackets
    fa, fb = f(a), f(b)
    if ((fa * fb) >= 0).any():
        raise ValueError("f(a) and f(b) must have opposite signs")

    # Initialize solver state in C++
    state = torch.ops.torchscience.brent_init(a, b, fa, fb, xtol, ftol)

    for _ in range(maxiter):
        # C++ returns next points to evaluate (or signals convergence)
        x_eval, converged = torch.ops.torchscience.brent_step(state)

        if converged:
            return torch.ops.torchscience.brent_result(state)

        # Evaluate f at requested points
        fx = f(x_eval)

        # Update state with new function values
        torch.ops.torchscience.brent_update(state, fx)

    raise RuntimeError(f"brent: failed to converge in {maxiter} iterations")
```

The C++ state object tracks per-element:
- Current bracket `[a, b]` and function values `f(a), f(b)`
- Best estimate and previous iterates (for interpolation)
- Convergence flags

## Autograd Implementation

Uses implicit differentiation via the implicit function theorem. For a root `x*` where `f(x*, θ) = 0`:

```
dx*/dθ = -[∂f/∂x]⁻¹ · ∂f/∂θ
```

The Python wrapper computes `∂f/∂x` at the root and attaches a gradient hook:

```python
def brent(f, a, b, ...):
    root = _brent_impl(f, a, b, ...)

    if root.requires_grad:
        with torch.enable_grad():
            x = root.detach().requires_grad_(True)
            fx = f(x)
            df_dx = torch.autograd.grad(fx.sum(), x)[0]

        root = _attach_implicit_grad(root, df_dx)

    return root
```

## Error Handling

### Validation

- Shape mismatch between `a` and `b`: `ValueError`
- Same sign at bracket endpoints: `ValueError` with count of invalid brackets
- `a == b` (degenerate bracket): `ValueError`
- NaN/Inf in inputs: `ValueError`

### Edge Cases

| Case | Behavior |
|------|----------|
| `f(a) == 0` | Return `a` immediately |
| `f(b) == 0` | Return `b` immediately |
| `f` returns NaN | `RuntimeError` with diagnostic |
| Non-convergence | `RuntimeError` after `maxiter` |

### Dtype-Aware Defaults

| Dtype | Default `xtol` | Default `ftol` |
|-------|----------------|----------------|
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-3 | 1e-3 |
| float32 | 1e-6 | 1e-6 |
| float64 | 1e-12 | 1e-12 |

## Testing Strategy

Test file: `tests/torchscience/root_finding/test__brent.py`

### Test Categories

1. **Basic functionality**: simple quadratic, batched roots, trigonometric functions
2. **Convergence**: verify `xtol` and `ftol` conditions, dtype-aware defaults
3. **Edge cases**: root at endpoint, invalid bracket raises, maxiter exceeded
4. **Autograd**: gradient flow, `gradcheck`, `gradgradcheck`
5. **Device/dtype**: CUDA support, float32/float64 precision

### Reference Comparisons

- Compare against `scipy.optimize.brentq` for correctness
- Verify batched results match individual scalar calls
