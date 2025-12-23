# Brent's Root-Finding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `torchscience.root_finding.brent` - a batched, differentiable Brent's root-finding method.

**Architecture:** Stateful iteration protocol where Python drives the loop (calling user's function `f`) and C++ handles the numerical Brent's method updates. Autograd via implicit differentiation computed in Python.

**Tech Stack:** PyTorch, C++17, CUDA (optional)

---

## Task 1: Create Python Module Structure

**Files:**
- Create: `src/torchscience/root_finding/__init__.py`
- Create: `src/torchscience/root_finding/_brent.py`
- Modify: `src/torchscience/__init__.py`

**Step 1: Create the root_finding module init**

```python
# src/torchscience/root_finding/__init__.py
from ._brent import brent

__all__ = ["brent"]
```

**Step 2: Create stub brent function**

```python
# src/torchscience/root_finding/_brent.py
from typing import Callable

import torch
from torch import Tensor


def _get_default_tol(dtype: torch.dtype) -> float:
    """Get dtype-aware default tolerance."""
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-3
    elif dtype == torch.float32:
        return 1e-6
    else:  # float64
        return 1e-12


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
        Tolerance on interval width. Default: dtype-aware.
    ftol : float, optional
        Tolerance on |f(x)|. Default: dtype-aware.
    maxiter : int
        Maximum iterations. Raises RuntimeError if exceeded.

    Returns
    -------
    Tensor
        Roots of shape (N,).
    """
    raise NotImplementedError("brent not yet implemented")
```

**Step 3: Export root_finding from main module**

Add to `src/torchscience/__init__.py`:

```python
from . import (
    _csrc,
    optimization,
    root_finding,  # Add this line
    signal_processing,
    statistics,
)

__all__ = [
    "_csrc",
    "optimization",
    "root_finding",  # Add this line
    "signal_processing",
    "statistics",
]
```

**Step 4: Verify import works**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run python -c "from torchscience.root_finding import brent; print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
git add src/torchscience/root_finding/ src/torchscience/__init__.py
git commit -m "feat: add root_finding module structure"
```

---

## Task 2: Create Test File with First Failing Test

**Files:**
- Create: `tests/torchscience/root_finding/__init__.py`
- Create: `tests/torchscience/root_finding/test__brent.py`

**Step 1: Create test directory init**

```python
# tests/torchscience/root_finding/__init__.py
```

**Step 2: Write first failing test**

```python
# tests/torchscience/root_finding/test__brent.py
import math

import pytest
import torch

from torchscience.root_finding import brent


class TestBrent:
    """Tests for Brent's root-finding method."""

    def test_simple_quadratic(self):
        """Find sqrt(2) by solving x^2 - 2 = 0."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root = brent(f, a, b)

        expected = math.sqrt(2)
        torch.testing.assert_close(root, torch.tensor([expected]), rtol=1e-6, atol=1e-6)
```

**Step 3: Run test to verify it fails**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run pytest tests/torchscience/root_finding/test__brent.py::TestBrent::test_simple_quadratic -v`

Expected: FAIL with `NotImplementedError: brent not yet implemented`

**Step 4: Commit failing test**

```bash
git add tests/torchscience/root_finding/
git commit -m "test: add first failing test for brent"
```

---

## Task 3: Implement Pure Python Brent's Method

**Files:**
- Modify: `src/torchscience/root_finding/_brent.py`

**Step 1: Implement validation and core algorithm**

Replace the `brent` function with:

```python
# src/torchscience/root_finding/_brent.py
from typing import Callable

import torch
from torch import Tensor


def _get_default_tol(dtype: torch.dtype) -> float:
    """Get dtype-aware default tolerance."""
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-3
    elif dtype == torch.float32:
        return 1e-6
    else:  # float64
        return 1e-12


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
        Tolerance on interval width. Default: dtype-aware.
    ftol : float, optional
        Tolerance on |f(x)|. Default: dtype-aware.
    maxiter : int
        Maximum iterations. Raises RuntimeError if exceeded.

    Returns
    -------
    Tensor
        Roots of shape (N,).

    Raises
    ------
    ValueError
        If f(a) and f(b) have the same sign for any element.
    RuntimeError
        If convergence is not achieved within maxiter iterations.
    """
    # Input validation
    if a.shape != b.shape:
        raise ValueError(f"a and b must have same shape, got {a.shape} and {b.shape}")

    if a.numel() == 0:
        return a.clone()

    # Flatten for processing, remember original shape
    orig_shape = a.shape
    a = a.flatten()
    b = b.flatten()

    # Get tolerances
    dtype = a.dtype
    if xtol is None:
        xtol = _get_default_tol(dtype)
    if ftol is None:
        ftol = _get_default_tol(dtype)

    # Evaluate function at endpoints
    fa = f(a)
    fb = f(b)

    # Check for valid brackets
    if torch.any(fa * fb >= 0):
        invalid = fa * fb >= 0
        raise ValueError(
            f"Invalid bracket: f(a) and f(b) must have opposite signs. "
            f"{invalid.sum().item()} of {invalid.numel()} brackets are invalid."
        )

    # Check for NaN/Inf in inputs
    if torch.any(~torch.isfinite(a)) or torch.any(~torch.isfinite(b)):
        raise ValueError("a and b must not contain NaN or Inf")

    # Check for roots at endpoints
    root = torch.where(fa == 0, a, torch.where(fb == 0, b, a.clone()))
    at_endpoint = (fa == 0) | (fb == 0)
    if torch.all(at_endpoint):
        return root.reshape(orig_shape)

    # Ensure |f(a)| >= |f(b)| by swapping if needed
    swap_mask = torch.abs(fa) < torch.abs(fb)
    a, b = torch.where(swap_mask, b, a), torch.where(swap_mask, a, b)
    fa, fb = torch.where(swap_mask, fb, fa), torch.where(swap_mask, fa, fb)

    # Initialize state
    c = a.clone()  # Previous iterate
    fc = fa.clone()
    d = b - a  # Step size
    e = d.clone()  # Previous step size

    # Track which elements have converged
    converged = at_endpoint.clone()
    result = root.clone()

    for iteration in range(maxiter):
        # Check convergence: both xtol AND ftol must be satisfied
        interval_small = torch.abs(b - a) < xtol
        residual_small = torch.abs(fb) < ftol
        newly_converged = interval_small & residual_small & ~converged
        converged = converged | newly_converged
        result = torch.where(newly_converged, b, result)

        if torch.all(converged):
            return result.reshape(orig_shape)

        # Only update unconverged elements
        active = ~converged

        # Brent's method update
        # If f(b) and f(c) have same sign, reset c to a
        same_sign = (fb * fc > 0) & active
        c = torch.where(same_sign, a, c)
        fc = torch.where(same_sign, fa, fc)
        d = torch.where(same_sign, b - a, d)
        e = torch.where(same_sign, d, e)

        # If |f(c)| < |f(b)|, swap b and c
        swap = (torch.abs(fc) < torch.abs(fb)) & active
        a = torch.where(swap, b, a)
        fa = torch.where(swap, fb, fa)
        b = torch.where(swap, c, b)
        fb = torch.where(swap, fc, fb)
        c = torch.where(swap, a, c)
        fc = torch.where(swap, fa, fc)

        # Compute tolerance
        tol = 2 * torch.finfo(dtype).eps * torch.abs(b) + xtol / 2
        m = (c - b) / 2

        # Check if bisection is needed
        use_bisection = (torch.abs(e) < tol) | (torch.abs(fa) <= torch.abs(fb))

        # Try interpolation
        s = fb / fa
        # Linear interpolation (secant) when a == c
        p_secant = 2 * m * s
        q_secant = 1 - s

        # Inverse quadratic interpolation when a != c
        r = fb / fc
        t = fa / fc
        p_iqp = s * (2 * m * t * (t - r) - (b - a) * (r - 1))
        q_iqp = (t - 1) * (r - 1) * (s - 1)

        use_secant = torch.abs(a - c) < tol
        p = torch.where(use_secant, p_secant, p_iqp)
        q = torch.where(use_secant, q_secant, q_iqp)

        # Ensure q > 0
        neg_q = q < 0
        p = torch.where(neg_q, -p, p)
        q = torch.where(neg_q, -q, q)

        # Accept interpolation if it's better than bisection
        accept_interp = (
            (2 * p < 3 * m * q - torch.abs(tol * q))
            & (p < torch.abs(e * q / 2))
            & ~use_bisection
        )

        # Update step
        e_new = torch.where(accept_interp, d, m)
        d_new = torch.where(accept_interp, p / q, m)

        e = torch.where(active, e_new, e)
        d = torch.where(active, d_new, d)

        # Move best guess to a, update b
        a = torch.where(active, b, a)
        fa = torch.where(active, fb, fa)

        # Compute new b
        step = torch.where(torch.abs(d) > tol, d, torch.sign(m) * tol)
        b = torch.where(active, b + step, b)

        # Evaluate function at new b
        fb_new = f(b)

        # Check for NaN in function evaluation
        if torch.any(torch.isnan(fb_new) & active):
            raise RuntimeError("Function returned NaN during iteration")

        fb = torch.where(active, fb_new, fb)

    raise RuntimeError(f"brent: failed to converge in {maxiter} iterations")
```

**Step 2: Run test to verify it passes**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run pytest tests/torchscience/root_finding/test__brent.py::TestBrent::test_simple_quadratic -v`

Expected: PASS

**Step 3: Commit**

```bash
git add src/torchscience/root_finding/_brent.py
git commit -m "feat: implement pure Python Brent's method"
```

---

## Task 4: Add Batched and Edge Case Tests

**Files:**
- Modify: `tests/torchscience/root_finding/test__brent.py`

**Step 1: Add more tests**

Add to test class:

```python
    def test_batched_roots(self):
        """Find multiple roots in parallel."""
        c = torch.tensor([2.0, 3.0, 4.0, 5.0])
        f = lambda x: x**2 - c
        a = torch.ones(4)
        b = torch.full((4,), 10.0)

        roots = brent(f, a, b)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)

    def test_trigonometric(self):
        """Find root of sin(x) = 0 in [2, 4] -> pi."""
        f = lambda x: torch.sin(x)
        a = torch.tensor([2.0])
        b = torch.tensor([4.0])

        root = brent(f, a, b)

        torch.testing.assert_close(root, torch.tensor([math.pi]), rtol=1e-6, atol=1e-6)

    def test_root_at_endpoint_a(self):
        """Return a immediately if f(a) == 0."""
        f = lambda x: x - 1.0
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root = brent(f, a, b)

        torch.testing.assert_close(root, torch.tensor([1.0]))

    def test_root_at_endpoint_b(self):
        """Return b immediately if f(b) == 0."""
        f = lambda x: x - 2.0
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root = brent(f, a, b)

        torch.testing.assert_close(root, torch.tensor([2.0]))

    def test_invalid_bracket_raises(self):
        """Raise ValueError when f(a) and f(b) have same sign."""
        f = lambda x: x**2 + 1  # Always positive
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        with pytest.raises(ValueError, match="Invalid bracket"):
            brent(f, a, b)

    def test_invalid_bracket_count(self):
        """Error message includes count of invalid brackets."""
        f = lambda x: x**2 - torch.tensor([2.0, -1.0, 3.0])  # 2nd is invalid
        a = torch.tensor([1.0, 1.0, 1.0])
        b = torch.tensor([2.0, 2.0, 2.0])

        with pytest.raises(ValueError, match="1 of 3 brackets are invalid"):
            brent(f, a, b)

    def test_shape_mismatch_raises(self):
        """Raise ValueError when a and b have different shapes."""
        f = lambda x: x
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0])

        with pytest.raises(ValueError, match="must have same shape"):
            brent(f, a, b)

    def test_nan_input_raises(self):
        """Raise ValueError when inputs contain NaN."""
        f = lambda x: x
        a = torch.tensor([float("nan")])
        b = torch.tensor([1.0])

        with pytest.raises(ValueError, match="must not contain NaN"):
            brent(f, a, b)

    def test_maxiter_exceeded_raises(self):
        """Raise RuntimeError when maxiter is exceeded."""
        # Function that converges very slowly
        f = lambda x: x**3 - x - 1
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        with pytest.raises(RuntimeError, match="failed to converge"):
            brent(f, a, b, maxiter=1)

    def test_float32(self):
        """Works correctly with float32."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0], dtype=torch.float32)
        b = torch.tensor([2.0], dtype=torch.float32)

        root = brent(f, a, b)

        assert root.dtype == torch.float32
        torch.testing.assert_close(root, torch.tensor([math.sqrt(2)], dtype=torch.float32), rtol=1e-5, atol=1e-5)

    def test_float64(self):
        """Works correctly with float64."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)

        root = brent(f, a, b)

        assert root.dtype == torch.float64
        torch.testing.assert_close(root, torch.tensor([math.sqrt(2)], dtype=torch.float64), rtol=1e-10, atol=1e-10)

    def test_convergence_xtol(self):
        """Verify interval width is within xtol at convergence."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        xtol = 1e-10

        root = brent(f, a, b, xtol=xtol)

        # The root should be accurate to xtol
        expected = math.sqrt(2)
        assert abs(root.item() - expected) < xtol * 10  # Some margin

    def test_convergence_ftol(self):
        """Verify |f(x)| is within ftol at convergence."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        ftol = 1e-12

        root = brent(f, a, b, ftol=ftol)

        residual = abs(f(root).item())
        assert residual < ftol * 10  # Some margin

    def test_preserves_shape(self):
        """Output has same shape as input."""
        f = lambda x: x**2 - 2
        a = torch.ones(2, 3)
        b = torch.full((2, 3), 2.0)

        root = brent(f.reshape(-1), a, b)

        assert root.shape == (2, 3)

    def test_empty_input(self):
        """Handle empty input gracefully."""
        f = lambda x: x
        a = torch.tensor([])
        b = torch.tensor([])

        root = brent(f, a, b)

        assert root.shape == (0,)
```

**Step 2: Run all tests**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run pytest tests/torchscience/root_finding/test__brent.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/torchscience/root_finding/test__brent.py
git commit -m "test: add comprehensive tests for brent"
```

---

## Task 5: Add Autograd Support via Implicit Differentiation

**Files:**
- Modify: `src/torchscience/root_finding/_brent.py`

**Step 1: Create custom autograd function**

Add before the `brent` function:

```python
class _BrentImplicitGrad(torch.autograd.Function):
    """Custom autograd for implicit differentiation through root-finding."""

    @staticmethod
    def forward(ctx, root: Tensor, df_dx: Tensor) -> Tensor:
        ctx.save_for_backward(df_dx)
        return root

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        (df_dx,) = ctx.saved_tensors
        # Implicit function theorem: dx/dtheta = -[df/dx]^{-1} * df/dtheta
        # Since we're differentiating x (the root), we have:
        # grad_input = grad_output / df_dx
        grad_root = grad_output / df_dx
        return grad_root, None
```

**Step 2: Modify brent to attach gradient**

Add at the end of `brent`, before the final return and after convergence:

```python
    # Attach implicit gradient if needed
    if result.requires_grad:
        # Compute df/dx at the root
        x = result.detach().requires_grad_(True)
        with torch.enable_grad():
            fx = f(x)
            df_dx = torch.autograd.grad(
                fx, x, grad_outputs=torch.ones_like(fx), create_graph=False
            )[0]
        result = _BrentImplicitGrad.apply(result, df_dx)

    return result.reshape(orig_shape)
```

Note: You'll need to restructure the return logic slightly to ensure this happens in all return paths.

**Step 3: Run autograd test**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run pytest tests/torchscience/root_finding/test__brent.py::TestBrent::test_gradient_simple -v`

Expected: PASS

**Step 4: Commit**

```bash
git add src/torchscience/root_finding/_brent.py
git commit -m "feat: add autograd support via implicit differentiation"
```

---

## Task 6: Add Autograd Tests

**Files:**
- Modify: `tests/torchscience/root_finding/test__brent.py`

**Step 1: Add gradient tests**

Add to test class:

```python
    def test_gradient_simple(self):
        """Gradient flows through root-finding."""
        # Solve x^2 - c = 0 for x, where c is a parameter
        c = torch.tensor([2.0], requires_grad=True)
        f = lambda x: x**2 - c
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root = brent(f, a, b)
        root.sum().backward()

        # d(sqrt(c))/dc = 1 / (2 * sqrt(c))
        expected_grad = 1 / (2 * torch.sqrt(c.detach()))
        torch.testing.assert_close(c.grad, expected_grad, rtol=1e-4, atol=1e-4)

    def test_gradient_batched(self):
        """Gradient works for batched inputs."""
        c = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
        f = lambda x: x**2 - c
        a = torch.ones(3)
        b = torch.full((3,), 10.0)

        roots = brent(f, a, b)
        roots.sum().backward()

        expected_grad = 1 / (2 * torch.sqrt(c.detach()))
        torch.testing.assert_close(c.grad, expected_grad, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self):
        """torch.autograd.gradcheck passes."""

        def fn(c):
            f = lambda x: x**2 - c
            a = torch.ones_like(c)
            b = torch.full_like(c, 10.0)
            return brent(f, a, b)

        c = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(fn, (c,), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_no_grad_when_not_required(self):
        """No gradient computation when requires_grad=False."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root = brent(f, a, b)

        assert not root.requires_grad
```

**Step 2: Run gradient tests**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run pytest tests/torchscience/root_finding/test__brent.py -k gradient -v`

Expected: All gradient tests PASS

**Step 3: Commit**

```bash
git add tests/torchscience/root_finding/test__brent.py
git commit -m "test: add autograd tests for brent"
```

---

## Task 7: Add CUDA Support Test

**Files:**
- Modify: `tests/torchscience/root_finding/test__brent.py`

**Step 1: Add CUDA test**

Add to test class:

```python
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Works on CUDA tensors."""
        c = torch.tensor([2.0, 3.0, 4.0], device="cuda")
        f = lambda x: x**2 - c
        a = torch.ones(3, device="cuda")
        b = torch.full((3,), 10.0, device="cuda")

        roots = brent(f, a, b)

        assert roots.device.type == "cuda"
        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_gradient(self):
        """Gradient works on CUDA."""
        c = torch.tensor([2.0, 3.0], device="cuda", requires_grad=True)
        f = lambda x: x**2 - c
        a = torch.ones(2, device="cuda")
        b = torch.full((2,), 10.0, device="cuda")

        roots = brent(f, a, b)
        roots.sum().backward()

        expected_grad = 1 / (2 * torch.sqrt(c.detach()))
        torch.testing.assert_close(c.grad, expected_grad, rtol=1e-4, atol=1e-4)
```

**Step 2: Run CUDA test (if available)**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run pytest tests/torchscience/root_finding/test__brent.py -k cuda -v`

Expected: PASS or SKIPPED if CUDA not available

**Step 3: Commit**

```bash
git add tests/torchscience/root_finding/test__brent.py
git commit -m "test: add CUDA tests for brent"
```

---

## Task 8: Add Comparison with SciPy

**Files:**
- Modify: `tests/torchscience/root_finding/test__brent.py`

**Step 1: Add scipy comparison test**

Add to test file (at top):

```python
scipy = pytest.importorskip("scipy")
from scipy.optimize import brentq as scipy_brentq
```

Add to test class:

```python
    def test_matches_scipy(self):
        """Results match scipy.optimize.brentq."""
        # Test several different functions
        test_cases = [
            (lambda x: x**2 - 2, 1.0, 2.0),  # sqrt(2)
            (lambda x: x**3 - x - 1, 1.0, 2.0),  # cubic
            (lambda x: torch.cos(x) if isinstance(x, Tensor) else math.cos(x), 0.0, 2.0),  # cos root
        ]

        for f_torch, a_val, b_val in test_cases:
            # Scipy version
            f_scipy = lambda x: f_torch(torch.tensor(x)).item() if hasattr(f_torch(torch.tensor(a_val)), 'item') else f_torch(x)
            scipy_root = scipy_brentq(lambda x: f_torch(torch.tensor([x])).item(), a_val, b_val)

            # Our version
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            our_root = brent(f_torch, a, b)

            torch.testing.assert_close(
                our_root,
                torch.tensor([scipy_root], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )
```

**Step 2: Run scipy test**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run pytest tests/torchscience/root_finding/test__brent.py::TestBrent::test_matches_scipy -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/torchscience/root_finding/test__brent.py
git commit -m "test: add scipy comparison for brent"
```

---

## Task 9: Add Docstring and Examples

**Files:**
- Modify: `src/torchscience/root_finding/_brent.py`

**Step 1: Expand docstring with examples**

Update the docstring of `brent` function:

```python
def brent(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    *,
    xtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 100,
) -> Tensor:
    r"""
    Find roots of f(x) = 0 using Brent's method.

    Brent's method combines bisection, secant, and inverse quadratic
    interpolation to find roots with superlinear convergence while
    maintaining the reliability of bisection.

    This implementation is batched: it finds roots for multiple
    bracketed intervals in parallel.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vectorized function. Takes tensor of shape (N,), returns (N,).
        The function should be continuous on each bracket [a_i, b_i].
    a, b : Tensor
        Bracket endpoints. Shape (N,) or broadcastable to (N,).
        Must satisfy f(a) * f(b) < 0 for each element.
    xtol : float, optional
        Tolerance on interval width. Convergence requires |b - a| < xtol.
        Default: 1e-3 for float16/bfloat16, 1e-6 for float32, 1e-12 for float64.
    ftol : float, optional
        Tolerance on residual. Convergence requires |f(x)| < ftol.
        Default: same as xtol.
    maxiter : int, default=100
        Maximum number of iterations.

    Returns
    -------
    Tensor
        Roots of shape matching input. Supports autograd via implicit
        differentiation.

    Raises
    ------
    ValueError
        If f(a) and f(b) have the same sign for any element, or if
        inputs have mismatched shapes or contain NaN/Inf.
    RuntimeError
        If convergence is not achieved within maxiter iterations,
        or if f returns NaN during iteration.

    Examples
    --------
    Find the square root of 2:

    >>> f = lambda x: x**2 - 2
    >>> a = torch.tensor([1.0])
    >>> b = torch.tensor([2.0])
    >>> root = brent(f, a, b)
    >>> root
    tensor([1.4142])

    Find multiple roots in parallel:

    >>> c = torch.tensor([2.0, 3.0, 4.0])
    >>> f = lambda x: x**2 - c
    >>> roots = brent(f, torch.ones(3), torch.full((3,), 10.0))
    >>> roots  # sqrt(2), sqrt(3), sqrt(4)
    tensor([1.4142, 1.7321, 2.0000])

    Differentiate through root-finding:

    >>> c = torch.tensor([2.0], requires_grad=True)
    >>> f = lambda x: x**2 - c
    >>> root = brent(f, torch.tensor([1.0]), torch.tensor([2.0]))
    >>> root.backward()
    >>> c.grad  # d(sqrt(c))/dc = 1/(2*sqrt(c))
    tensor([0.3536])

    Notes
    -----
    Convergence requires BOTH xtol AND ftol conditions to be satisfied.
    This is more conservative than scipy.optimize.brentq (which uses
    only xtol) but ensures both the interval is narrow and the residual
    is small.

    Autograd is implemented via implicit differentiation using the
    implicit function theorem. At a root x* where f(x*, θ) = 0:

    .. math::

        \frac{dx^*}{d\theta} = -\left[\frac{\partial f}{\partial x}\right]^{-1}
        \frac{\partial f}{\partial \theta}

    This is memory-efficient (doesn't store iteration history) and
    numerically stable.

    See Also
    --------
    scipy.optimize.brentq : SciPy's scalar Brent's method
    """
```

**Step 2: Test doctests work**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run python -m doctest src/torchscience/root_finding/_brent.py -v`

Expected: Examples run without error (may need to adjust output format)

**Step 3: Commit**

```bash
git add src/torchscience/root_finding/_brent.py
git commit -m "docs: add comprehensive docstring for brent"
```

---

## Task 10: Run Full Test Suite and Final Cleanup

**Files:**
- None (verification only)

**Step 1: Run full test suite**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run pytest tests/ -v --tb=short`

Expected: All tests pass (including existing tests)

**Step 2: Run linting**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run ruff check src/torchscience/root_finding/ tests/torchscience/root_finding/`

Expected: No errors

**Step 3: Format code**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/brent-root-finding && uv run ruff format src/torchscience/root_finding/ tests/torchscience/root_finding/`

**Step 4: Commit any formatting changes**

```bash
git add -A
git commit -m "style: format root_finding module" || echo "No formatting changes"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | Create Python module structure | Import verification |
| 2 | First failing test | `test_simple_quadratic` fails |
| 3 | Implement pure Python Brent | `test_simple_quadratic` passes |
| 4 | Comprehensive tests | All edge cases |
| 5 | Add autograd support | Gradient flows |
| 6 | Autograd tests | `gradcheck` passes |
| 7 | CUDA tests | Works on GPU |
| 8 | SciPy comparison | Matches reference |
| 9 | Documentation | Docstring complete |
| 10 | Final verification | Full suite passes |

**Note:** The C++/CUDA backend optimization (as described in the design doc) is deferred to a future iteration. The pure Python implementation is functionally complete and leverages PyTorch's tensor operations for parallelism.
