# torchscience.spline Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement differentiable spline interpolation for PyTorch tensors with cubic splines and B-splines.

**Architecture:** Pure Python implementation using tensorclass data containers. Tridiagonal solver via Thomas algorithm, B-spline basis via Cox-de Boor recursion. All operations written as pure functions for vmap/jacrev compatibility.

**Tech Stack:** PyTorch, tensordict (tensorclass), scipy (test comparisons only)

---

## Phase 1: Module Structure and Exceptions

### Task 1.1: Create module directory structure

**Files:**
- Create: `src/torchscience/spline/__init__.py`
- Create: `src/torchscience/spline/_exceptions.py`
- Create: `tests/torchscience/spline/__init__.py`

**Step 1: Create the spline module directory**

```bash
mkdir -p src/torchscience/spline
mkdir -p tests/torchscience/spline
```

**Step 2: Write the exceptions module**

```python
# src/torchscience/spline/_exceptions.py
"""Exceptions for the spline module."""


class SplineError(Exception):
    """Base exception for spline operations."""

    pass


class ExtrapolationError(SplineError):
    """Raised when query point is outside spline domain with extrapolate='error'."""

    pass


class KnotError(SplineError):
    """Raised for invalid knot vectors (non-monotonic, insufficient knots)."""

    pass


class DegreeError(SplineError):
    """Raised when degree is invalid for given knot count."""

    pass
```

**Step 3: Write the __init__.py files**

```python
# src/torchscience/spline/__init__.py
"""Differentiable spline interpolation for PyTorch tensors.

This module provides cubic splines and B-splines with full autograd support.

Cubic Splines
-------------
cubic_spline_fit
    Fit a cubic spline to data points.
cubic_spline_evaluate
    Evaluate a cubic spline at query points.
cubic_spline_derivative
    Compute derivatives of a cubic spline.
cubic_spline_integral
    Compute definite integral of a cubic spline.
cubic_spline_interpolate
    Convenience function: fit + evaluate in one call.

B-Splines
---------
b_spline
    Construct a B-spline from knots and control points.
b_spline_fit
    Fit a B-spline to data points.
b_spline_evaluate
    Evaluate a B-spline at query points.
b_spline_derivative
    Compute derivatives of a B-spline.
b_spline_basis
    Evaluate B-spline basis functions.

Data Types
----------
CubicSpline
    Piecewise cubic polynomial interpolant.
BSpline
    B-spline curve.

Exceptions
----------
SplineError
    Base exception for spline operations.
ExtrapolationError
    Query point outside spline domain.
KnotError
    Invalid knot vector.
DegreeError
    Invalid degree for given knots.
"""

from torchscience.spline._exceptions import (
    DegreeError,
    ExtrapolationError,
    KnotError,
    SplineError,
)

__all__ = [
    # Exceptions
    "SplineError",
    "ExtrapolationError",
    "KnotError",
    "DegreeError",
]
```

```python
# tests/torchscience/spline/__init__.py
"""Tests for torchscience.spline module."""
```

**Step 4: Verify imports work**

Run: `uv run python -c "from torchscience.spline import SplineError, ExtrapolationError, KnotError, DegreeError; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add src/torchscience/spline/ tests/torchscience/spline/
git commit -m "feat(spline): add module structure and exceptions"
```

---

## Phase 2: Tridiagonal Solver

### Task 2.1: Write failing test for tridiagonal solver

**Files:**
- Create: `tests/torchscience/spline/test__tridiagonal.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/spline/test__tridiagonal.py
"""Tests for tridiagonal solver."""

import pytest
import torch


class TestSolveTridiagonal:
    def test_simple_3x3_system(self):
        """Test solving a simple 3x3 tridiagonal system."""
        from torchscience.spline._tridiagonal import solve_tridiagonal

        # System: [2 1 0] [x0]   [1]
        #         [1 2 1] [x1] = [2]
        #         [0 1 2] [x2]   [1]
        diag = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        upper = torch.tensor([1.0, 1.0], dtype=torch.float64)
        lower = torch.tensor([1.0, 1.0], dtype=torch.float64)
        rhs = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)

        x = solve_tridiagonal(diag, upper, lower, rhs)

        # Verify Ax = b
        result = torch.zeros_like(rhs)
        result[0] = diag[0] * x[0] + upper[0] * x[1]
        result[1] = lower[0] * x[0] + diag[1] * x[1] + upper[1] * x[2]
        result[2] = lower[1] * x[1] + diag[2] * x[2]

        torch.testing.assert_close(result, rhs, rtol=1e-10, atol=1e-10)

    def test_batched_rhs(self):
        """Test with batched right-hand side."""
        from torchscience.spline._tridiagonal import solve_tridiagonal

        diag = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        upper = torch.tensor([1.0, 1.0], dtype=torch.float64)
        lower = torch.tensor([1.0, 1.0], dtype=torch.float64)
        # Batch of 2 right-hand sides, each of length 3
        rhs = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0]], dtype=torch.float64)

        x = solve_tridiagonal(diag, upper, lower, rhs)

        assert x.shape == (2, 3)
        # Second solution should be 2x the first
        torch.testing.assert_close(x[1], 2 * x[0], rtol=1e-10, atol=1e-10)

    def test_gradcheck(self):
        """Test gradients through the solver."""
        from torchscience.spline._tridiagonal import solve_tridiagonal

        diag = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, requires_grad=True)
        upper = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
        lower = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
        rhs = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64, requires_grad=True)

        def fn(d, u, l, r):
            return solve_tridiagonal(d, u, l, r)

        assert torch.autograd.gradcheck(fn, (diag, upper, lower, rhs), eps=1e-6)

    def test_larger_system(self):
        """Test a larger system against torch.linalg.solve."""
        from torchscience.spline._tridiagonal import solve_tridiagonal

        n = 50
        diag = 4 * torch.ones(n, dtype=torch.float64)
        upper = torch.ones(n - 1, dtype=torch.float64)
        lower = torch.ones(n - 1, dtype=torch.float64)
        rhs = torch.randn(n, dtype=torch.float64)

        x = solve_tridiagonal(diag, upper, lower, rhs)

        # Build full matrix and solve with torch.linalg.solve
        A = torch.diag(diag) + torch.diag(upper, 1) + torch.diag(lower, -1)
        x_expected = torch.linalg.solve(A, rhs)

        torch.testing.assert_close(x, x_expected, rtol=1e-10, atol=1e-10)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/torchscience/spline/test__tridiagonal.py -v`
Expected: FAIL with "cannot import name 'solve_tridiagonal'"

**Step 3: Commit failing test**

```bash
git add tests/torchscience/spline/test__tridiagonal.py
git commit -m "test(spline): add failing tests for tridiagonal solver"
```

### Task 2.2: Implement tridiagonal solver

**Files:**
- Create: `src/torchscience/spline/_tridiagonal.py`

**Step 1: Write the implementation**

```python
# src/torchscience/spline/_tridiagonal.py
"""Tridiagonal matrix solver using Thomas algorithm."""

import torch
from torch import Tensor


def solve_tridiagonal(
    diag: Tensor,
    upper: Tensor,
    lower: Tensor,
    rhs: Tensor,
) -> Tensor:
    """
    Solve a tridiagonal system Ax = b using the Thomas algorithm.

    The matrix A has the form:
        [d0  u0   0   0  ...  0   0 ]
        [l0  d1  u1   0  ...  0   0 ]
        [ 0  l1  d2  u2  ...  0   0 ]
        [        ...                ]
        [ 0   0   0   0  ... ln-2 dn-1]

    Parameters
    ----------
    diag : Tensor
        Main diagonal, shape (n,)
    upper : Tensor
        Upper diagonal, shape (n-1,)
    lower : Tensor
        Lower diagonal, shape (n-1,)
    rhs : Tensor
        Right-hand side, shape (*batch, n)

    Returns
    -------
    Tensor
        Solution x, shape (*batch, n)

    Notes
    -----
    This implementation is fully differentiable. The backward pass
    solves another tridiagonal system (the adjoint) with O(n) complexity.
    """
    n = diag.shape[0]

    # Move system axis to front for easier indexing
    # rhs: (*batch, n) -> (n, *batch)
    rhs_t = rhs.movedim(-1, 0)
    batch_shape = rhs_t.shape[1:]

    # Forward elimination (modify copies to preserve gradient graph)
    c_prime = torch.zeros(n - 1, dtype=diag.dtype, device=diag.device)
    d_prime = torch.zeros((n,) + batch_shape, dtype=rhs.dtype, device=rhs.device)

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs_t[0] / diag[0]

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs_t[i] - lower[i - 1] * d_prime[i - 1]) / denom

    # Last row (no upper diagonal)
    denom = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
    d_prime[n - 1] = (rhs_t[n - 1] - lower[n - 2] * d_prime[n - 2]) / denom

    # Back substitution
    x = torch.zeros_like(d_prime)
    x[n - 1] = d_prime[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    # Move system axis back: (n, *batch) -> (*batch, n)
    return x.movedim(0, -1)
```

**Step 2: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/spline/test__tridiagonal.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/torchscience/spline/_tridiagonal.py
git commit -m "feat(spline): implement tridiagonal solver with Thomas algorithm"
```

---

## Phase 3: CubicSpline Tensorclass and Fitting

### Task 3.1: Write failing test for CubicSpline tensorclass

**Files:**
- Create: `tests/torchscience/spline/test__cubic_spline.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/spline/test__cubic_spline.py
"""Tests for cubic spline interpolation."""

import math

import pytest
import torch


class TestCubicSplineFit:
    def test_fit_returns_cubic_spline(self):
        """Test that cubic_spline_fit returns a CubicSpline tensorclass."""
        from torchscience.spline import CubicSpline, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y)

        assert isinstance(spline, CubicSpline)
        assert spline.knots.shape == (5,)
        assert spline.coefficients.shape == (4, 4)  # 4 segments, 4 coeffs each

    def test_fit_natural_boundary(self):
        """Test natural boundary conditions (zero second derivative at ends)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3  # Known cubic

        spline = cubic_spline_fit(x, y, boundary="natural")

        # For natural splines, second derivative at endpoints should be ~0
        # Coefficient layout: [a, b, c, d] for a + b*(t-ti) + c*(t-ti)^2 + d*(t-ti)^3
        # Second derivative at ti is 2*c
        # Check first segment at x[0]
        assert abs(spline.coefficients[0, 2].item()) < 1e-6

    def test_fit_not_a_knot_boundary(self):
        """Test not-a-knot boundary (default)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 2 * math.pi, 10, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y, boundary="not_a_knot")

        assert spline.boundary == "not_a_knot"

    def test_fit_clamped_boundary(self):
        """Test clamped boundary conditions (known first derivatives)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        # dy/dx = 2x, so at x=0: 0, at x=1: 2
        boundary_values = torch.tensor([0.0, 2.0], dtype=torch.float64)

        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        assert spline.boundary == "clamped"

    def test_fit_periodic_boundary(self):
        """Test periodic boundary conditions."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 2 * math.pi, 10, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y, boundary="periodic")

        assert spline.boundary == "periodic"

    def test_fit_validates_monotonic_knots(self):
        """Test that non-monotonic knots raise KnotError."""
        from torchscience.spline import KnotError, cubic_spline_fit

        x = torch.tensor([0.0, 0.5, 0.3, 1.0], dtype=torch.float64)  # Not monotonic
        y = torch.tensor([0.0, 0.5, 0.3, 1.0], dtype=torch.float64)

        with pytest.raises(KnotError):
            cubic_spline_fit(x, y)

    def test_fit_validates_minimum_points(self):
        """Test that too few points raises KnotError."""
        from torchscience.spline import KnotError, cubic_spline_fit

        x = torch.tensor([0.0], dtype=torch.float64)
        y = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(KnotError):
            cubic_spline_fit(x, y)

    def test_fit_multidimensional_values(self):
        """Test fitting with multi-dimensional y values (e.g., 3D curve)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack([torch.sin(x), torch.cos(x), x], dim=-1)  # (5, 3)

        spline = cubic_spline_fit(x, y)

        # 4 segments, 4 coefficients, 3 value dimensions
        assert spline.coefficients.shape == (4, 4, 3)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/torchscience/spline/test__cubic_spline.py::TestCubicSplineFit -v`
Expected: FAIL with "cannot import name 'CubicSpline'"

**Step 3: Commit failing test**

```bash
git add tests/torchscience/spline/test__cubic_spline.py
git commit -m "test(spline): add failing tests for CubicSpline fit"
```

### Task 3.2: Implement CubicSpline tensorclass

**Files:**
- Create: `src/torchscience/spline/_cubic_spline.py`
- Modify: `src/torchscience/spline/__init__.py`

**Step 1: Write the CubicSpline tensorclass**

```python
# src/torchscience/spline/_cubic_spline.py
"""Cubic spline interpolation."""

from typing import Optional

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.spline._exceptions import KnotError
from torchscience.spline._tridiagonal import solve_tridiagonal


@tensorclass
class CubicSpline:
    """Piecewise cubic polynomial interpolant.

    Attributes
    ----------
    knots : Tensor
        Breakpoints, shape (n_knots,). Strictly increasing.
    coefficients : Tensor
        Polynomial coefficients, shape (n_segments, 4, *value_shape).
        For segment i, the polynomial is:
        a[i] + b[i]*(t-knots[i]) + c[i]*(t-knots[i])^2 + d[i]*(t-knots[i])^3
        where coefficients[i] = [a, b, c, d].
    boundary : str
        Boundary condition type: "natural", "clamped", "not_a_knot", "periodic".
    extrapolate : str
        Extrapolation mode: "error", "clamp", "extrapolate".
    """

    knots: Tensor
    coefficients: Tensor
    boundary: str
    extrapolate: str


def cubic_spline_fit(
    x: Tensor,
    y: Tensor,
    boundary: str = "not_a_knot",
    boundary_values: Optional[Tensor] = None,
    extrapolate: str = "error",
) -> CubicSpline:
    """
    Fit a cubic spline to data points.

    Parameters
    ----------
    x : Tensor
        Knot positions, shape (n_points,). Must be strictly increasing.
    y : Tensor
        Values at knots, shape (n_points, *value_shape).
    boundary : str
        Boundary condition: "natural", "clamped", "not_a_knot", "periodic".
    boundary_values : Tensor, optional
        For "clamped": first derivatives at endpoints, shape (2, *value_shape).
    extrapolate : str
        Extrapolation mode: "error", "clamp", "extrapolate".

    Returns
    -------
    CubicSpline
        Fitted spline.

    Raises
    ------
    KnotError
        If x is not strictly increasing or has fewer than 2 points.
    """
    n = x.shape[0]

    # Validate knots
    if n < 2:
        raise KnotError(f"Need at least 2 points, got {n}")
    if not torch.all(x[1:] > x[:-1]):
        raise KnotError("Knots must be strictly increasing")

    # Compute interval widths
    h = x[1:] - x[:-1]  # (n-1,)

    # Get value shape
    if y.dim() == 1:
        value_shape = ()
        y_flat = y
    else:
        value_shape = y.shape[1:]
        # Flatten value dimensions for computation
        y_flat = y.reshape(n, -1)  # (n, prod(value_shape))

    n_values = y_flat.shape[1] if y_flat.dim() > 1 else 1
    if y_flat.dim() == 1:
        y_flat = y_flat.unsqueeze(-1)

    # Compute second differences for RHS
    # delta[i] = (y[i+1] - y[i]) / h[i]
    delta = (y_flat[1:] - y_flat[:-1]) / h.unsqueeze(-1)  # (n-1, n_values)

    # Build tridiagonal system for second derivatives m
    # Interior equations: h[i-1]*m[i-1] + 2*(h[i-1]+h[i])*m[i] + h[i]*m[i+1] = 6*(delta[i] - delta[i-1])

    if boundary == "natural":
        # Natural: m[0] = m[n-1] = 0
        # Solve reduced (n-2) x (n-2) system for m[1:n-1]
        if n == 2:
            # Only one segment, no interior points
            m = torch.zeros(2, n_values, dtype=y.dtype, device=y.device)
        else:
            diag = 2 * (h[:-1] + h[1:])  # (n-2,)
            upper = h[1:-1]  # (n-3,)
            lower = h[1:-1]  # (n-3,)
            rhs = 6 * (delta[1:] - delta[:-1])  # (n-2, n_values)

            if n == 3:
                # 1x1 system
                m_interior = rhs / diag.unsqueeze(-1)
            else:
                m_interior = solve_tridiagonal(diag, upper, lower, rhs.T).T

            # Pad with zeros at endpoints
            m = torch.cat(
                [
                    torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
                    m_interior,
                    torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
                ],
                dim=0,
            )

    elif boundary == "clamped":
        if boundary_values is None:
            raise ValueError("boundary_values required for clamped boundary")
        bv = boundary_values.reshape(2, -1)  # (2, n_values)

        # Full n x n system
        diag = torch.zeros(n, dtype=x.dtype, device=x.device)
        diag[0] = 2 * h[0]
        diag[1:-1] = 2 * (h[:-1] + h[1:])
        diag[-1] = 2 * h[-1]

        upper = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        upper[0] = h[0]
        upper[1:] = h[1:]

        lower = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        lower[:-1] = h[:-1]
        lower[-1] = h[-1]

        rhs = torch.zeros(n, n_values, dtype=y.dtype, device=y.device)
        rhs[0] = 6 * (delta[0] - bv[0])
        rhs[1:-1] = 6 * (delta[1:] - delta[:-1])
        rhs[-1] = 6 * (bv[1] - delta[-1])

        m = solve_tridiagonal(diag, upper, lower, rhs.T).T

    elif boundary == "not_a_knot":
        if n < 4:
            # Fall back to natural for small systems
            return cubic_spline_fit(x, y, boundary="natural", extrapolate=extrapolate)

        # Not-a-knot: third derivative continuous at x[1] and x[n-2]
        # This modifies the first and last equations
        diag = torch.zeros(n, dtype=x.dtype, device=x.device)
        upper = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        lower = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        rhs = torch.zeros(n, n_values, dtype=y.dtype, device=y.device)

        # First equation: not-a-knot at x[1]
        diag[0] = h[1]
        upper[0] = -(h[0] + h[1])
        rhs[0] = 0

        # Interior equations
        for i in range(1, n - 1):
            diag[i] = 2 * (h[i - 1] + h[i])
            if i < n - 1:
                upper[i] = h[i]
            if i > 0:
                lower[i - 1] = h[i - 1]
            rhs[i] = 6 * (delta[i] - delta[i - 1]) if i < n - 1 else torch.zeros(n_values)

        # Last equation: not-a-knot at x[n-2]
        lower[-1] = -(h[-2] + h[-1])
        diag[-1] = h[-2]
        rhs[-1] = 0

        # Fix interior equations RHS
        rhs[1:-1] = 6 * (delta[1:] - delta[:-1])

        m = solve_tridiagonal(diag, upper, lower, rhs.T).T

    elif boundary == "periodic":
        if not torch.allclose(y_flat[0], y_flat[-1]):
            raise KnotError("For periodic boundary, y[0] must equal y[-1]")

        # Periodic boundary uses Sherman-Morrison or cyclic reduction
        # For simplicity, use natural boundary on cyclic extension
        # (Production code would use Sherman-Morrison for O(n) solve)
        # Solve reduced system m[0] = m[n-1]
        diag = 2 * (h[:-1] + h[1:])  # (n-2,)
        if n > 3:
            upper = h[1:-1].clone()
            lower = h[1:-1].clone()
        else:
            upper = torch.tensor([], dtype=x.dtype, device=x.device)
            lower = torch.tensor([], dtype=x.dtype, device=x.device)
        rhs = 6 * (delta[1:] - delta[:-1])

        if n == 3:
            m_interior = rhs / diag.unsqueeze(-1)
        elif n > 3:
            m_interior = solve_tridiagonal(diag, upper, lower, rhs.T).T
        else:
            m_interior = torch.zeros(0, n_values, dtype=y.dtype, device=y.device)

        # For periodic, set m[0] = m[-1]
        m = torch.cat(
            [
                torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
                m_interior,
                torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
            ],
            dim=0,
        )
        m[-1] = m[0]  # Enforce periodicity

    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")

    # Compute polynomial coefficients for each segment
    # p_i(t) = a_i + b_i*(t-x_i) + c_i*(t-x_i)^2 + d_i*(t-x_i)^3
    # where:
    #   a_i = y_i
    #   b_i = delta_i - h_i * (2*m_i + m_{i+1}) / 6
    #   c_i = m_i / 2
    #   d_i = (m_{i+1} - m_i) / (6 * h_i)

    n_seg = n - 1
    a = y_flat[:-1]  # (n_seg, n_values)
    c = m[:-1] / 2  # (n_seg, n_values)
    d = (m[1:] - m[:-1]) / (6 * h.unsqueeze(-1))  # (n_seg, n_values)
    b = delta - h.unsqueeze(-1) * (2 * m[:-1] + m[1:]) / 6  # (n_seg, n_values)

    # Stack coefficients: (n_seg, 4, n_values)
    coeffs = torch.stack([a, b, c, d], dim=1)

    # Reshape back to original value shape
    if value_shape:
        coeffs = coeffs.reshape(n_seg, 4, *value_shape)

    return CubicSpline(
        knots=x,
        coefficients=coeffs,
        boundary=boundary,
        extrapolate=extrapolate,
        batch_size=[],
    )
```

**Step 2: Update __init__.py**

```python
# src/torchscience/spline/__init__.py
"""Differentiable spline interpolation for PyTorch tensors.

...existing docstring...
"""

from torchscience.spline._cubic_spline import CubicSpline, cubic_spline_fit
from torchscience.spline._exceptions import (
    DegreeError,
    ExtrapolationError,
    KnotError,
    SplineError,
)

__all__ = [
    # Data types
    "CubicSpline",
    # Cubic spline functions
    "cubic_spline_fit",
    # Exceptions
    "SplineError",
    "ExtrapolationError",
    "KnotError",
    "DegreeError",
]
```

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/spline/test__cubic_spline.py::TestCubicSplineFit -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/spline/_cubic_spline.py src/torchscience/spline/__init__.py
git commit -m "feat(spline): implement CubicSpline tensorclass and fitting"
```

---

## Phase 4: CubicSpline Evaluation

### Task 4.1: Write failing test for cubic_spline_evaluate

**Files:**
- Modify: `tests/torchscience/spline/test__cubic_spline.py`

**Step 1: Add tests for evaluation**

```python
# Add to tests/torchscience/spline/test__cubic_spline.py

class TestCubicSplineEvaluate:
    def test_evaluate_at_knots(self):
        """Test that evaluation at knots returns original values."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y)
        y_interp = cubic_spline_evaluate(spline, x)

        torch.testing.assert_close(y_interp, y, rtol=1e-10, atol=1e-10)

    def test_evaluate_between_knots(self):
        """Test evaluation at points between knots."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 2 * math.pi, 20, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y)

        # Evaluate at midpoints
        t = (x[:-1] + x[1:]) / 2
        y_interp = cubic_spline_evaluate(spline, t)
        y_expected = torch.sin(t)

        # Should be close to true sine
        torch.testing.assert_close(y_interp, y_expected, rtol=1e-4, atol=1e-4)

    def test_evaluate_scalar_query(self):
        """Test evaluation at a single point."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        spline = cubic_spline_fit(x, y)
        t = torch.tensor(0.5, dtype=torch.float64)
        y_interp = cubic_spline_evaluate(spline, t)

        assert y_interp.shape == ()
        torch.testing.assert_close(y_interp, torch.tensor(0.25, dtype=torch.float64), rtol=1e-3, atol=1e-3)

    def test_evaluate_extrapolate_error(self):
        """Test that extrapolation raises error by default."""
        from torchscience.spline import (
            ExtrapolationError,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x

        spline = cubic_spline_fit(x, y, extrapolate="error")
        t = torch.tensor(1.5, dtype=torch.float64)

        with pytest.raises(ExtrapolationError):
            cubic_spline_evaluate(spline, t)

    def test_evaluate_extrapolate_clamp(self):
        """Test clamp extrapolation mode."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x

        spline = cubic_spline_fit(x, y, extrapolate="clamp")

        t_below = torch.tensor(-0.5, dtype=torch.float64)
        t_above = torch.tensor(1.5, dtype=torch.float64)

        y_below = cubic_spline_evaluate(spline, t_below)
        y_above = cubic_spline_evaluate(spline, t_above)

        # Should clamp to endpoint values
        torch.testing.assert_close(y_below, y[0], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(y_above, y[-1], rtol=1e-10, atol=1e-10)

    def test_evaluate_extrapolate_extend(self):
        """Test extrapolate mode that extends the polynomial."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        spline = cubic_spline_fit(x, y, extrapolate="extrapolate")

        t = torch.tensor(1.5, dtype=torch.float64)
        y_interp = cubic_spline_evaluate(spline, t)

        # Should extrapolate using the last segment's polynomial
        assert y_interp.shape == ()

    def test_evaluate_multidimensional(self):
        """Test evaluation with multi-dimensional values."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack([x, x**2, x**3], dim=-1)  # (5, 3)

        spline = cubic_spline_fit(x, y)
        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        y_interp = cubic_spline_evaluate(spline, t)

        assert y_interp.shape == (3, 3)  # (n_query, value_dim)

    def test_gradcheck(self):
        """Test gradients through evaluation."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)
        y.requires_grad_(True)

        spline = cubic_spline_fit(x, y)
        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True)

        def fn(t_in):
            return cubic_spline_evaluate(spline, t_in)

        assert torch.autograd.gradcheck(fn, (t,), eps=1e-6)

    def test_scipy_comparison(self):
        """Compare against scipy.interpolate.CubicSpline."""
        pytest.importorskip("scipy")
        from scipy.interpolate import CubicSpline as ScipyCubicSpline

        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 2 * math.pi, 10, dtype=torch.float64)
        y = torch.sin(x)

        # Fit with torchscience
        spline = cubic_spline_fit(x, y, boundary="natural")
        t = torch.linspace(0, 2 * math.pi, 100, dtype=torch.float64)
        y_torch = cubic_spline_evaluate(spline, t)

        # Fit with scipy
        scipy_spline = ScipyCubicSpline(x.numpy(), y.numpy(), bc_type="natural")
        y_scipy = torch.from_numpy(scipy_spline(t.numpy()))

        torch.testing.assert_close(y_torch, y_scipy, rtol=1e-10, atol=1e-10)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/torchscience/spline/test__cubic_spline.py::TestCubicSplineEvaluate::test_evaluate_at_knots -v`
Expected: FAIL with "cannot import name 'cubic_spline_evaluate'"

**Step 3: Commit failing test**

```bash
git add tests/torchscience/spline/test__cubic_spline.py
git commit -m "test(spline): add failing tests for cubic_spline_evaluate"
```

### Task 4.2: Implement cubic_spline_evaluate

**Files:**
- Modify: `src/torchscience/spline/_cubic_spline.py`
- Modify: `src/torchscience/spline/__init__.py`

**Step 1: Add evaluation function**

```python
# Add to src/torchscience/spline/_cubic_spline.py

from torchscience.spline._exceptions import ExtrapolationError, KnotError


def cubic_spline_evaluate(spline: CubicSpline, t: Tensor) -> Tensor:
    """
    Evaluate a cubic spline at query points.

    Parameters
    ----------
    spline : CubicSpline
        Fitted cubic spline.
    t : Tensor
        Query points, shape (*query_shape,).

    Returns
    -------
    Tensor
        Interpolated values, shape (*query_shape, *value_shape).

    Raises
    ------
    ExtrapolationError
        If t is outside [knots[0], knots[-1]] and extrapolate="error".
    """
    knots = spline.knots
    coeffs = spline.coefficients
    extrapolate = spline.extrapolate

    t_flat = t.reshape(-1)
    n_query = t_flat.shape[0]

    # Check bounds
    t_min, t_max = knots[0], knots[-1]

    if extrapolate == "error":
        if torch.any(t_flat < t_min) or torch.any(t_flat > t_max):
            raise ExtrapolationError(
                f"Query points outside domain [{t_min.item()}, {t_max.item()}]"
            )
        t_clamped = t_flat
    elif extrapolate == "clamp":
        t_clamped = torch.clamp(t_flat, t_min, t_max)
    elif extrapolate == "extrapolate":
        t_clamped = t_flat
    else:
        raise ValueError(f"Unknown extrapolate mode: {extrapolate}")

    # Find segment indices using searchsorted
    # segment_idx[i] is the segment containing t_clamped[i]
    segment_idx = torch.searchsorted(knots[1:-1], t_clamped)

    # Handle extrapolation
    if extrapolate == "extrapolate":
        segment_idx = torch.clamp(segment_idx, 0, len(knots) - 2)

    # Compute local coordinate within segment
    t_local = t_clamped - knots[segment_idx]

    # Get coefficients for each query point
    # coeffs: (n_segments, 4, *value_shape)
    # We need coeffs[segment_idx] for each query point
    a = coeffs[segment_idx, 0]  # (n_query, *value_shape)
    b = coeffs[segment_idx, 1]
    c = coeffs[segment_idx, 2]
    d = coeffs[segment_idx, 3]

    # Evaluate polynomial: a + b*dt + c*dt^2 + d*dt^3
    # Broadcast t_local to match value shape
    dt = t_local
    for _ in range(a.dim() - 1):
        dt = dt.unsqueeze(-1)

    result = a + b * dt + c * dt**2 + d * dt**3

    # Reshape to match input query shape
    output_shape = t.shape + coeffs.shape[2:]
    return result.reshape(output_shape)
```

**Step 2: Update __init__.py**

```python
# Add to src/torchscience/spline/__init__.py imports and __all__

from torchscience.spline._cubic_spline import (
    CubicSpline,
    cubic_spline_evaluate,
    cubic_spline_fit,
)

__all__ = [
    # Data types
    "CubicSpline",
    # Cubic spline functions
    "cubic_spline_fit",
    "cubic_spline_evaluate",
    # Exceptions
    ...
]
```

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/spline/test__cubic_spline.py::TestCubicSplineEvaluate -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/spline/_cubic_spline.py src/torchscience/spline/__init__.py
git commit -m "feat(spline): implement cubic_spline_evaluate"
```

---

## Phase 5: CubicSpline Derivatives and Integral

### Task 5.1: Implement cubic_spline_derivative

**Files:**
- Modify: `src/torchscience/spline/_cubic_spline.py`
- Modify: `tests/torchscience/spline/test__cubic_spline.py`

**Step 1: Write failing test**

```python
# Add to tests/torchscience/spline/test__cubic_spline.py

class TestCubicSplineDerivative:
    def test_derivative_of_linear(self):
        """Test derivative of linear function is constant."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = 2 * x + 1  # dy/dx = 2

        spline = cubic_spline_fit(x, y)
        t = torch.linspace(0, 1, 10, dtype=torch.float64)
        dy = cubic_spline_derivative(spline, t)

        torch.testing.assert_close(
            dy, 2 * torch.ones_like(t), rtol=1e-6, atol=1e-6
        )

    def test_derivative_of_quadratic(self):
        """Test derivative of quadratic function."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = x**2  # dy/dx = 2x

        spline = cubic_spline_fit(x, y)
        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        dy = cubic_spline_derivative(spline, t)

        expected = 2 * t
        torch.testing.assert_close(dy, expected, rtol=1e-3, atol=1e-3)

    def test_second_derivative(self):
        """Test second derivative."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = x**3  # d2y/dx2 = 6x

        spline = cubic_spline_fit(x, y)
        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        d2y = cubic_spline_derivative(spline, t, order=2)

        expected = 6 * t
        torch.testing.assert_close(d2y, expected, rtol=1e-2, atol=1e-2)

    def test_derivative_multidimensional(self):
        """Test derivative with multi-dimensional values."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.stack([x, x**2], dim=-1)  # (10, 2)

        spline = cubic_spline_fit(x, y)
        t = torch.tensor([0.5], dtype=torch.float64)
        dy = cubic_spline_derivative(spline, t)

        # dy/dx = [1, 2x] at x=0.5: [1, 1]
        expected = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        torch.testing.assert_close(dy, expected, rtol=1e-3, atol=1e-3)
```

**Step 2: Run to verify failure, then implement**

```python
# Add to src/torchscience/spline/_cubic_spline.py

def cubic_spline_derivative(
    spline: CubicSpline,
    t: Tensor,
    order: int = 1,
) -> Tensor:
    """
    Compute derivative of cubic spline at query points.

    Parameters
    ----------
    spline : CubicSpline
        Fitted cubic spline.
    t : Tensor
        Query points, shape (*query_shape,).
    order : int
        Derivative order (1, 2, or 3).

    Returns
    -------
    Tensor
        Derivative values, shape (*query_shape, *value_shape).
    """
    if order < 1 or order > 3:
        raise ValueError(f"Derivative order must be 1, 2, or 3, got {order}")

    knots = spline.knots
    coeffs = spline.coefficients
    extrapolate = spline.extrapolate

    t_flat = t.reshape(-1)
    t_min, t_max = knots[0], knots[-1]

    if extrapolate == "error":
        if torch.any(t_flat < t_min) or torch.any(t_flat > t_max):
            raise ExtrapolationError(
                f"Query points outside domain [{t_min.item()}, {t_max.item()}]"
            )
        t_clamped = t_flat
    elif extrapolate == "clamp":
        t_clamped = torch.clamp(t_flat, t_min, t_max)
    else:
        t_clamped = t_flat

    segment_idx = torch.searchsorted(knots[1:-1], t_clamped)
    if extrapolate == "extrapolate":
        segment_idx = torch.clamp(segment_idx, 0, len(knots) - 2)

    t_local = t_clamped - knots[segment_idx]

    # Get coefficients
    b = coeffs[segment_idx, 1]
    c = coeffs[segment_idx, 2]
    d = coeffs[segment_idx, 3]

    dt = t_local
    for _ in range(b.dim() - 1):
        dt = dt.unsqueeze(-1)

    if order == 1:
        # d/dt(a + b*t + c*t^2 + d*t^3) = b + 2*c*t + 3*d*t^2
        result = b + 2 * c * dt + 3 * d * dt**2
    elif order == 2:
        # d2/dt2 = 2*c + 6*d*t
        result = 2 * c + 6 * d * dt
    else:  # order == 3
        # d3/dt3 = 6*d
        result = 6 * d

    output_shape = t.shape + coeffs.shape[2:]
    return result.reshape(output_shape)
```

**Step 3: Run tests, commit**

Run: `uv run python -m pytest tests/torchscience/spline/test__cubic_spline.py::TestCubicSplineDerivative -v`

```bash
git add src/torchscience/spline/_cubic_spline.py tests/torchscience/spline/test__cubic_spline.py
git commit -m "feat(spline): implement cubic_spline_derivative"
```

### Task 5.2: Implement cubic_spline_integral

**Step 1: Write failing test**

```python
# Add to tests/torchscience/spline/test__cubic_spline.py

class TestCubicSplineIntegral:
    def test_integral_of_constant(self):
        """Test integral of constant function."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = 2 * torch.ones_like(x)  # f(x) = 2

        spline = cubic_spline_fit(x, y)
        integral = cubic_spline_integral(spline, 0.0, 1.0)

        # Integral of 2 from 0 to 1 is 2
        torch.testing.assert_close(
            integral, torch.tensor(2.0, dtype=torch.float64), rtol=1e-6, atol=1e-6
        )

    def test_integral_of_linear(self):
        """Test integral of linear function."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x  # f(x) = x

        spline = cubic_spline_fit(x, y)
        integral = cubic_spline_integral(spline, 0.0, 1.0)

        # Integral of x from 0 to 1 is 0.5
        torch.testing.assert_close(
            integral, torch.tensor(0.5, dtype=torch.float64), rtol=1e-6, atol=1e-6
        )

    def test_integral_partial_domain(self):
        """Test integral over partial domain."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        x = torch.linspace(0, 2, 10, dtype=torch.float64)
        y = x**2  # f(x) = x^2

        spline = cubic_spline_fit(x, y)
        integral = cubic_spline_integral(spline, 0.5, 1.5)

        # Integral of x^2 from 0.5 to 1.5 is [x^3/3]_0.5^1.5 = 1.125 - 0.0417 = 1.0833...
        expected = (1.5**3 - 0.5**3) / 3
        torch.testing.assert_close(
            integral, torch.tensor(expected, dtype=torch.float64), rtol=1e-3, atol=1e-3
        )
```

**Step 2: Implement**

```python
# Add to src/torchscience/spline/_cubic_spline.py

def cubic_spline_integral(
    spline: CubicSpline,
    a: float,
    b: float,
) -> Tensor:
    """
    Compute definite integral of cubic spline over [a, b].

    Parameters
    ----------
    spline : CubicSpline
        Fitted cubic spline.
    a : float
        Lower integration bound.
    b : float
        Upper integration bound.

    Returns
    -------
    Tensor
        Integral value, shape (*value_shape,).
    """
    knots = spline.knots
    coeffs = spline.coefficients

    # Find segments that overlap with [a, b]
    a_t = torch.tensor(a, dtype=knots.dtype, device=knots.device)
    b_t = torch.tensor(b, dtype=knots.dtype, device=knots.device)

    # Clamp to spline domain
    a_clamped = torch.clamp(a_t, knots[0], knots[-1])
    b_clamped = torch.clamp(b_t, knots[0], knots[-1])

    if a_clamped >= b_clamped:
        return torch.zeros(coeffs.shape[2:], dtype=coeffs.dtype, device=coeffs.device)

    # Find first and last segments
    seg_start = torch.searchsorted(knots[1:-1], a_clamped).item()
    seg_end = torch.searchsorted(knots[1:-1], b_clamped).item()

    total = torch.zeros(coeffs.shape[2:], dtype=coeffs.dtype, device=coeffs.device)

    for seg in range(seg_start, seg_end + 1):
        # Integration bounds for this segment
        t0 = max(a_clamped.item(), knots[seg].item())
        t1 = min(b_clamped.item(), knots[seg + 1].item())

        if t0 >= t1:
            continue

        # Local coordinates
        dt0 = t0 - knots[seg].item()
        dt1 = t1 - knots[seg].item()

        # Get coefficients
        a_coef = coeffs[seg, 0]
        b_coef = coeffs[seg, 1]
        c_coef = coeffs[seg, 2]
        d_coef = coeffs[seg, 3]

        # Integral of a + b*t + c*t^2 + d*t^3 is:
        # a*t + b*t^2/2 + c*t^3/3 + d*t^4/4
        def antiderivative(dt):
            return (
                a_coef * dt
                + b_coef * dt**2 / 2
                + c_coef * dt**3 / 3
                + d_coef * dt**4 / 4
            )

        total = total + antiderivative(dt1) - antiderivative(dt0)

    return total
```

**Step 3: Run tests, commit**

Run: `uv run python -m pytest tests/torchscience/spline/test__cubic_spline.py::TestCubicSplineIntegral -v`

```bash
git add src/torchscience/spline/_cubic_spline.py tests/torchscience/spline/test__cubic_spline.py
git commit -m "feat(spline): implement cubic_spline_integral"
```

---

## Phase 6: B-Spline Basis Functions

### Task 6.1: Implement b_spline_basis with Cox-de Boor

**Files:**
- Create: `src/torchscience/spline/_basis.py`
- Create: `tests/torchscience/spline/test__basis.py`

**Step 1: Write failing test**

```python
# tests/torchscience/spline/test__basis.py
"""Tests for B-spline basis functions."""

import pytest
import torch


class TestBSplineBasis:
    def test_partition_of_unity(self):
        """Test that basis functions sum to 1."""
        from torchscience.spline._basis import b_spline_basis

        # Uniform knots for degree 3
        knots = torch.linspace(0, 1, 8, dtype=torch.float64)  # 8 knots
        t = torch.linspace(0.1, 0.9, 20, dtype=torch.float64)

        basis = b_spline_basis(knots, t, degree=3)

        # Sum over basis functions should be 1
        sums = basis.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-10, atol=1e-10)

    def test_non_negativity(self):
        """Test that basis functions are non-negative."""
        from torchscience.spline._basis import b_spline_basis

        knots = torch.linspace(0, 1, 10, dtype=torch.float64)
        t = torch.linspace(0.05, 0.95, 50, dtype=torch.float64)

        basis = b_spline_basis(knots, t, degree=3)

        assert torch.all(basis >= -1e-10)

    def test_local_support(self):
        """Test that each basis function has local support."""
        from torchscience.spline._basis import b_spline_basis

        knots = torch.linspace(0, 1, 10, dtype=torch.float64)
        degree = 3

        # Query at a single point
        t = torch.tensor([0.5], dtype=torch.float64)
        basis = b_spline_basis(knots, t, degree=degree)

        # At most degree+1 basis functions should be non-zero
        n_nonzero = (basis[0] > 1e-10).sum()
        assert n_nonzero <= degree + 1

    def test_degree_0(self):
        """Test degree-0 (piecewise constant) basis."""
        from torchscience.spline._basis import b_spline_basis

        knots = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        t = torch.tensor([0.25, 0.75], dtype=torch.float64)

        basis = b_spline_basis(knots, t, degree=0)

        # At t=0.25, only first basis function is 1
        torch.testing.assert_close(basis[0], torch.tensor([1.0, 0.0], dtype=torch.float64))
        # At t=0.75, only second basis function is 1
        torch.testing.assert_close(basis[1], torch.tensor([0.0, 1.0], dtype=torch.float64))

    def test_gradcheck_wrt_t(self):
        """Test gradients w.r.t. query points."""
        from torchscience.spline._basis import b_spline_basis

        knots = torch.linspace(0, 1, 8, dtype=torch.float64)
        t = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True)

        def fn(t_in):
            return b_spline_basis(knots, t_in, degree=3)

        assert torch.autograd.gradcheck(fn, (t,), eps=1e-6)

    def test_gradcheck_wrt_knots(self):
        """Test gradients w.r.t. knots (learnable knot placement)."""
        from torchscience.spline._basis import b_spline_basis

        knots = torch.linspace(0, 1, 8, dtype=torch.float64)
        knots.requires_grad_(True)
        t = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        def fn(k):
            return b_spline_basis(k, t, degree=3)

        assert torch.autograd.gradcheck(fn, (knots,), eps=1e-6)
```

**Step 2: Implement**

```python
# src/torchscience/spline/_basis.py
"""B-spline basis function evaluation using Cox-de Boor recursion."""

import torch
from torch import Tensor


def b_spline_basis(
    knots: Tensor,
    t: Tensor,
    degree: int,
) -> Tensor:
    """
    Evaluate B-spline basis functions at query points.

    Uses the Cox-de Boor recursion formula:
    N_{i,0}(t) = 1 if knots[i] <= t < knots[i+1], else 0
    N_{i,p}(t) = (t - knots[i]) / (knots[i+p] - knots[i]) * N_{i,p-1}(t)
                + (knots[i+p+1] - t) / (knots[i+p+1] - knots[i+1]) * N_{i+1,p-1}(t)

    Parameters
    ----------
    knots : Tensor
        Knot vector, shape (n_knots,). Must be non-decreasing.
    t : Tensor
        Query points, shape (n_query,).
    degree : int
        B-spline degree (0 for piecewise constant, 3 for cubic).

    Returns
    -------
    Tensor
        Basis function values, shape (n_query, n_basis) where
        n_basis = n_knots - degree - 1.
    """
    n_knots = knots.shape[0]
    n_basis = n_knots - degree - 1
    n_query = t.shape[0]

    if n_basis <= 0:
        raise ValueError(
            f"Need at least {degree + 2} knots for degree {degree}, got {n_knots}"
        )

    # Initialize degree-0 basis functions
    # N_{i,0}(t) = 1 if knots[i] <= t < knots[i+1], else 0
    # Shape: (n_query, n_knots - 1)
    t_expanded = t.unsqueeze(-1)  # (n_query, 1)
    knots_lo = knots[:-1].unsqueeze(0)  # (1, n_knots - 1)
    knots_hi = knots[1:].unsqueeze(0)  # (1, n_knots - 1)

    # Use soft indicator for differentiability
    # In practice, we use hard indicator for degree-0, then recursion is smooth
    N = ((t_expanded >= knots_lo) & (t_expanded < knots_hi)).to(t.dtype)

    # Handle right endpoint: include t == knots[-1] in last interval
    at_right_end = (t_expanded == knots[-1]).squeeze(-1)
    if at_right_end.any():
        N[at_right_end, -1] = 1.0

    # Cox-de Boor recursion
    for p in range(1, degree + 1):
        N_new = torch.zeros(n_query, n_knots - p - 1, dtype=t.dtype, device=t.device)

        for i in range(n_knots - p - 1):
            # Left term: (t - knots[i]) / (knots[i+p] - knots[i]) * N_{i,p-1}(t)
            denom_left = knots[i + p] - knots[i]
            if denom_left > 0:
                left = (t - knots[i]) / denom_left * N[:, i]
            else:
                left = torch.zeros_like(t)

            # Right term: (knots[i+p+1] - t) / (knots[i+p+1] - knots[i+1]) * N_{i+1,p-1}(t)
            denom_right = knots[i + p + 1] - knots[i + 1]
            if denom_right > 0:
                right = (knots[i + p + 1] - t) / denom_right * N[:, i + 1]
            else:
                right = torch.zeros_like(t)

            N_new[:, i] = left + right

        N = N_new

    return N
```

**Step 3: Run tests, commit**

Run: `uv run python -m pytest tests/torchscience/spline/test__basis.py -v`

```bash
git add src/torchscience/spline/_basis.py tests/torchscience/spline/test__basis.py
git commit -m "feat(spline): implement B-spline basis via Cox-de Boor recursion"
```

---

## Phase 7: BSpline Tensorclass and Evaluation

### Task 7.1: Implement BSpline tensorclass and evaluation

**Files:**
- Create: `src/torchscience/spline/_b_spline.py`
- Create: `tests/torchscience/spline/test__b_spline.py`

**Step 1: Write failing test**

```python
# tests/torchscience/spline/test__b_spline.py
"""Tests for B-spline curves."""

import pytest
import torch


class TestBSpline:
    def test_construction(self):
        """Test basic B-spline construction."""
        from torchscience.spline import BSpline, b_spline

        knots = torch.linspace(0, 1, 8, dtype=torch.float64)
        control_points = torch.randn(4, dtype=torch.float64)  # n_knots - degree - 1

        spline = b_spline(knots, control_points, degree=3)

        assert isinstance(spline, BSpline)
        assert spline.degree == 3

    def test_evaluate_endpoints(self):
        """Test that B-spline passes through endpoints for clamped knots."""
        from torchscience.spline import b_spline, b_spline_evaluate

        # Clamped knot vector: repeated knots at ends
        knots = torch.tensor(
            [0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=torch.float64
        )  # degree 3
        control_points = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], dtype=torch.float64)

        spline = b_spline(knots, control_points, degree=3)

        t_start = torch.tensor([0.0], dtype=torch.float64)
        t_end = torch.tensor([1.0], dtype=torch.float64)

        y_start = b_spline_evaluate(spline, t_start)
        y_end = b_spline_evaluate(spline, t_end)

        # For clamped B-splines, curve passes through first and last control points
        torch.testing.assert_close(y_start, control_points[:1], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(y_end, control_points[-1:], rtol=1e-10, atol=1e-10)

    def test_evaluate_2d_curve(self):
        """Test B-spline curve in 2D (parametric curve)."""
        from torchscience.spline import b_spline, b_spline_evaluate

        knots = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float64)
        # Control points for a Bezier curve (special case of B-spline)
        control_points = torch.tensor(
            [[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0]], dtype=torch.float64
        )

        spline = b_spline(knots, control_points, degree=3)

        t = torch.linspace(0, 1, 10, dtype=torch.float64)
        curve = b_spline_evaluate(spline, t)

        assert curve.shape == (10, 2)

    def test_learnable_control_points_gradient(self):
        """Test gradients w.r.t. control points."""
        from torchscience.spline import b_spline, b_spline_evaluate

        knots = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float64)
        control_points = torch.randn(4, dtype=torch.float64, requires_grad=True)

        def fn(cp):
            spline = b_spline(knots, cp, degree=3)
            t = torch.tensor([0.5], dtype=torch.float64)
            return b_spline_evaluate(spline, t)

        assert torch.autograd.gradcheck(fn, (control_points,), eps=1e-6)

    def test_learnable_knots_gradient(self):
        """Test gradients w.r.t. knots (learnable knot placement)."""
        from torchscience.spline import b_spline, b_spline_evaluate

        # Interior knots only (clamped ends fixed)
        interior_knots = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        control_points = torch.randn(4, dtype=torch.float64)

        def fn(ik):
            # Build full clamped knot vector
            knots = torch.cat([
                torch.zeros(4, dtype=torch.float64),
                ik,
                torch.ones(4, dtype=torch.float64),
            ])
            spline = b_spline(knots, control_points, degree=3)
            t = torch.tensor([0.5], dtype=torch.float64)
            return b_spline_evaluate(spline, t)

        assert torch.autograd.gradcheck(fn, (interior_knots,), eps=1e-6)
```

**Step 2: Implement**

```python
# src/torchscience/spline/_b_spline.py
"""B-spline curves."""

from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.spline._basis import b_spline_basis
from torchscience.spline._exceptions import DegreeError, ExtrapolationError


@tensorclass
class BSpline:
    """B-spline curve.

    Attributes
    ----------
    knots : Tensor
        Knot vector, shape (n_knots,).
    control_points : Tensor
        Control points, shape (n_control, *value_shape).
    degree : int
        Polynomial degree.
    extrapolate : str
        Extrapolation mode.
    """

    knots: Tensor
    control_points: Tensor
    degree: int
    extrapolate: str


def b_spline(
    knots: Tensor,
    control_points: Tensor,
    degree: int = 3,
    extrapolate: str = "error",
) -> BSpline:
    """
    Construct a B-spline from knots and control points.

    Parameters
    ----------
    knots : Tensor
        Knot vector, shape (n_knots,).
    control_points : Tensor
        Control points, shape (n_control, *value_shape).
    degree : int
        Polynomial degree.
    extrapolate : str
        Extrapolation mode.

    Returns
    -------
    BSpline
        B-spline curve.
    """
    n_knots = knots.shape[0]
    n_control = control_points.shape[0]
    expected_control = n_knots - degree - 1

    if n_control != expected_control:
        raise DegreeError(
            f"Expected {expected_control} control points for {n_knots} knots "
            f"and degree {degree}, got {n_control}"
        )

    return BSpline(
        knots=knots,
        control_points=control_points,
        degree=degree,
        extrapolate=extrapolate,
        batch_size=[],
    )


def b_spline_evaluate(spline: BSpline, t: Tensor) -> Tensor:
    """
    Evaluate B-spline at query points.

    Parameters
    ----------
    spline : BSpline
        B-spline curve.
    t : Tensor
        Query points, shape (*query_shape,).

    Returns
    -------
    Tensor
        Values at query points, shape (*query_shape, *value_shape).
    """
    knots = spline.knots
    control_points = spline.control_points
    degree = spline.degree
    extrapolate = spline.extrapolate

    t_flat = t.reshape(-1)

    # Get valid domain (from first to last non-repeated knot)
    t_min = knots[degree]
    t_max = knots[-degree - 1]

    if extrapolate == "error":
        if t_flat.min() < t_min or t_flat.max() > t_max:
            raise ExtrapolationError(
                f"Query points outside domain [{t_min.item()}, {t_max.item()}]"
            )
    elif extrapolate == "clamp":
        t_flat = t_flat.clamp(t_min, t_max)

    # Evaluate basis functions: (n_query, n_basis)
    basis = b_spline_basis(knots, t_flat, degree)

    # Compute weighted sum: result = sum_i N_i(t) * P_i
    # basis: (n_query, n_control)
    # control_points: (n_control, *value_shape)
    # result: (n_query, *value_shape)
    result = torch.einsum("qi,i...->q...", basis, control_points)

    # Reshape to match input query shape
    output_shape = t.shape + control_points.shape[1:]
    return result.reshape(output_shape)


# Need to import torch for einsum
import torch
```

**Step 3: Update __init__.py, run tests, commit**

```python
# Add to src/torchscience/spline/__init__.py

from torchscience.spline._b_spline import BSpline, b_spline, b_spline_evaluate
from torchscience.spline._basis import b_spline_basis

__all__ = [
    # Data types
    "CubicSpline",
    "BSpline",
    # Cubic spline functions
    "cubic_spline_fit",
    "cubic_spline_evaluate",
    "cubic_spline_derivative",
    "cubic_spline_integral",
    # B-spline functions
    "b_spline",
    "b_spline_evaluate",
    "b_spline_basis",
    # Exceptions
    ...
]
```

Run: `uv run python -m pytest tests/torchscience/spline/test__b_spline.py -v`

```bash
git add src/torchscience/spline/_b_spline.py src/torchscience/spline/__init__.py tests/torchscience/spline/test__b_spline.py
git commit -m "feat(spline): implement BSpline tensorclass and evaluation"
```

---

## Phase 8: B-Spline Fitting and Derivatives

### Task 8.1: Implement b_spline_fit

**Step 1: Write failing test**

```python
# Add to tests/torchscience/spline/test__b_spline.py

class TestBSplineFit:
    def test_fit_interpolates_data(self):
        """Test that fitted B-spline interpolates data points."""
        from torchscience.spline import b_spline_evaluate, b_spline_fit

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.sin(2 * 3.14159 * x)

        spline = b_spline_fit(x, y, n_control=6, degree=3)

        y_interp = b_spline_evaluate(spline, x)

        # Should interpolate closely (not exactly for least-squares fit)
        torch.testing.assert_close(y_interp, y, rtol=1e-2, atol=1e-2)

    def test_fit_custom_knots(self):
        """Test fitting with custom knot vector."""
        from torchscience.spline import b_spline_fit

        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        y = x**2

        # Custom non-uniform knots (more resolution at start)
        knots = torch.tensor(
            [0, 0, 0, 0, 0.1, 0.2, 0.5, 1, 1, 1, 1], dtype=torch.float64
        )

        spline = b_spline_fit(x, y, n_control=7, degree=3, knots=knots)

        assert spline.knots.shape[0] == 11
```

**Step 2: Implement**

```python
# Add to src/torchscience/spline/_b_spline.py

def b_spline_fit(
    x: Tensor,
    y: Tensor,
    n_control: int,
    degree: int = 3,
    knots: Tensor | None = None,
) -> BSpline:
    """
    Fit a B-spline to data points using least squares.

    Parameters
    ----------
    x : Tensor
        Data point locations, shape (n_points,).
    y : Tensor
        Data point values, shape (n_points, *value_shape).
    n_control : int
        Number of control points.
    degree : int
        B-spline degree.
    knots : Tensor, optional
        Custom knot vector. If None, uniform knots are generated.

    Returns
    -------
    BSpline
        Fitted B-spline.
    """
    n_points = x.shape[0]

    # Generate knots if not provided
    if knots is None:
        n_knots = n_control + degree + 1
        # Clamped uniform knots
        n_interior = n_knots - 2 * (degree + 1)
        if n_interior < 0:
            n_interior = 0
        interior = torch.linspace(
            x[0], x[-1], n_interior + 2, dtype=x.dtype, device=x.device
        )[1:-1]
        knots = torch.cat([
            x[0].expand(degree + 1),
            interior,
            x[-1].expand(degree + 1),
        ])

    # Evaluate basis at data points: (n_points, n_control)
    basis = b_spline_basis(knots, x, degree)

    # Solve least squares: basis @ control_points = y
    # Use torch.linalg.lstsq for batched least squares
    if y.dim() == 1:
        y_flat = y.unsqueeze(-1)
    else:
        y_flat = y.reshape(n_points, -1)

    # lstsq expects (m, n) @ (n, k) = (m, k)
    solution = torch.linalg.lstsq(basis, y_flat).solution

    # Reshape control points
    if y.dim() == 1:
        control_points = solution.squeeze(-1)
    else:
        control_points = solution.reshape(n_control, *y.shape[1:])

    return BSpline(
        knots=knots,
        control_points=control_points,
        degree=degree,
        extrapolate="error",
        batch_size=[],
    )
```

**Step 3: Run tests, commit**

```bash
git add src/torchscience/spline/_b_spline.py tests/torchscience/spline/test__b_spline.py
git commit -m "feat(spline): implement b_spline_fit via least squares"
```

### Task 8.2: Implement b_spline_derivative

**Step 1: Write failing test**

```python
# Add to tests/torchscience/spline/test__b_spline.py

class TestBSplineDerivative:
    def test_derivative_linear(self):
        """Test derivative of linear B-spline."""
        from torchscience.spline import b_spline, b_spline_derivative

        # Linear B-spline: straight line from (0,0) to (1,1)
        knots = torch.tensor([0, 0, 1, 1], dtype=torch.float64)  # degree 1
        control_points = torch.tensor([0.0, 1.0], dtype=torch.float64)

        spline = b_spline(knots, control_points, degree=1)
        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        dy = b_spline_derivative(spline, t)

        # Derivative of line y=x is 1
        expected = torch.ones_like(t)
        torch.testing.assert_close(dy, expected, rtol=1e-10, atol=1e-10)

    def test_derivative_quadratic(self):
        """Test derivative of quadratic B-spline."""
        from torchscience.spline import b_spline, b_spline_derivative, b_spline_fit

        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        y = x**2

        spline = b_spline_fit(x, y, n_control=5, degree=2)
        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        dy = b_spline_derivative(spline, t)

        # dy/dx of x^2 is 2x
        expected = 2 * t
        torch.testing.assert_close(dy, expected, rtol=1e-2, atol=1e-2)
```

**Step 2: Implement**

```python
# Add to src/torchscience/spline/_b_spline.py

def b_spline_derivative(
    spline: BSpline,
    t: Tensor,
    order: int = 1,
) -> Tensor:
    """
    Compute derivative of B-spline at query points.

    Uses the derivative formula: derivative of degree-p B-spline is
    a degree-(p-1) B-spline with differenced control points.

    Parameters
    ----------
    spline : BSpline
        B-spline curve.
    t : Tensor
        Query points.
    order : int
        Derivative order.

    Returns
    -------
    Tensor
        Derivative values.
    """
    knots = spline.knots
    control_points = spline.control_points
    degree = spline.degree

    if order > degree:
        # Derivative of polynomial beyond its degree is zero
        return torch.zeros(
            t.shape + control_points.shape[1:],
            dtype=control_points.dtype,
            device=control_points.device,
        )

    # Apply derivative formula iteratively
    d_knots = knots
    d_cp = control_points
    d_degree = degree

    for _ in range(order):
        n_cp = d_cp.shape[0]
        # New control points: P'_i = (p / (t_{i+p+1} - t_{i+1})) * (P_{i+1} - P_i)
        new_cp = []
        for i in range(n_cp - 1):
            denom = d_knots[i + d_degree + 1] - d_knots[i + 1]
            if denom > 0:
                coef = d_degree / denom
            else:
                coef = 0.0
            diff = d_cp[i + 1] - d_cp[i]
            new_cp.append(coef * diff)

        d_cp = torch.stack(new_cp)
        d_degree = d_degree - 1
        # Remove one knot from each end for derivative
        d_knots = d_knots[1:-1]

    # Evaluate the derivative B-spline
    deriv_spline = BSpline(
        knots=d_knots,
        control_points=d_cp,
        degree=d_degree,
        extrapolate=spline.extrapolate,
        batch_size=[],
    )

    return b_spline_evaluate(deriv_spline, t)
```

**Step 3: Run tests, commit**

```bash
git add src/torchscience/spline/_b_spline.py tests/torchscience/spline/test__b_spline.py
git commit -m "feat(spline): implement b_spline_derivative"
```

---

## Phase 9: Convenience Functions and Final Integration

### Task 9.1: Add cubic_spline_interpolate convenience function

```python
# Add to src/torchscience/spline/_cubic_spline.py

def cubic_spline_interpolate(
    x: Tensor,
    y: Tensor,
    t: Tensor,
    boundary: str = "not_a_knot",
    boundary_values: Tensor | None = None,
    extrapolate: str = "error",
) -> Tensor:
    """
    Fit and evaluate cubic spline in one call.

    Convenience function that combines cubic_spline_fit and cubic_spline_evaluate.

    Parameters
    ----------
    x : Tensor
        Knot positions.
    y : Tensor
        Values at knots.
    t : Tensor
        Query points.
    boundary : str
        Boundary condition.
    boundary_values : Tensor, optional
        For clamped boundary.
    extrapolate : str
        Extrapolation mode.

    Returns
    -------
    Tensor
        Interpolated values at query points.
    """
    spline = cubic_spline_fit(x, y, boundary, boundary_values, extrapolate)
    return cubic_spline_evaluate(spline, t)
```

### Task 9.2: Update __init__.py with all exports

```python
# Final src/torchscience/spline/__init__.py

"""Differentiable spline interpolation for PyTorch tensors.

This module provides cubic splines and B-splines with full autograd support.
"""

from torchscience.spline._b_spline import (
    BSpline,
    b_spline,
    b_spline_derivative,
    b_spline_evaluate,
    b_spline_fit,
)
from torchscience.spline._basis import b_spline_basis
from torchscience.spline._cubic_spline import (
    CubicSpline,
    cubic_spline_derivative,
    cubic_spline_evaluate,
    cubic_spline_fit,
    cubic_spline_integral,
    cubic_spline_interpolate,
)
from torchscience.spline._exceptions import (
    DegreeError,
    ExtrapolationError,
    KnotError,
    SplineError,
)

__all__ = [
    # Data types
    "CubicSpline",
    "BSpline",
    # Cubic spline functions
    "cubic_spline_fit",
    "cubic_spline_evaluate",
    "cubic_spline_derivative",
    "cubic_spline_integral",
    "cubic_spline_interpolate",
    # B-spline functions
    "b_spline",
    "b_spline_fit",
    "b_spline_evaluate",
    "b_spline_derivative",
    "b_spline_basis",
    # Exceptions
    "SplineError",
    "ExtrapolationError",
    "KnotError",
    "DegreeError",
]
```

### Task 9.3: Run full test suite and commit

Run: `uv run python -m pytest tests/torchscience/spline/ -v`

```bash
git add src/torchscience/spline/ tests/torchscience/spline/
git commit -m "feat(spline): complete spline module with cubic and B-spline support"
```

---

## Phase 10: Batching Tests

### Task 10.1: Add comprehensive batching tests

**Files:**
- Create: `tests/torchscience/spline/test__batching.py`

```python
# tests/torchscience/spline/test__batching.py
"""Tests for batched spline operations."""

import pytest
import torch
from torch.func import jacrev, vmap


class TestBatchedCubicSpline:
    def test_batched_fit_shared_knots(self):
        """Test fitting multiple curves with shared knots."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        # Batch of 5 curves
        y = torch.stack([torch.sin(x), torch.cos(x), x, x**2, x**3])  # (5, 10)

        spline = cubic_spline_fit(x, y)
        assert spline.coefficients.shape[0] == 5  # 5 batched splines

    def test_batched_evaluate(self):
        """Test evaluating batched spline at shared query points."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.stack([torch.sin(x), torch.cos(x)])  # (2, 10)

        spline = cubic_spline_fit(x, y)
        t = torch.linspace(0, 1, 20, dtype=torch.float64)
        result = cubic_spline_evaluate(spline, t)

        assert result.shape == (2, 20)

    def test_vmap_per_sample_queries(self):
        """Test vmap for per-sample query points."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.randn(3, 10, dtype=torch.float64)

        spline = cubic_spline_fit(x, y)

        # Each sample has its own query points
        t_batch = torch.rand(3, 5, dtype=torch.float64)

        # Use vmap to evaluate each spline at its own queries
        result = vmap(cubic_spline_evaluate)(spline, t_batch)

        assert result.shape == (3, 5)

    def test_vmap_jacobian(self):
        """Test batched Jacobian computation."""
        from torchscience.spline import b_spline, b_spline_evaluate

        knots = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float64)
        control_points = torch.randn(4, 2, dtype=torch.float64)  # 2D curve

        spline = b_spline(knots, control_points, degree=3)

        def eval_fn(t):
            return b_spline_evaluate(spline, t.unsqueeze(0)).squeeze(0)

        t = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        # Compute Jacobian of curve w.r.t. t
        jacobian = vmap(jacrev(eval_fn))(t)

        assert jacobian.shape == (3, 2)  # (n_query, value_dim)


class TestBatchedBSpline:
    def test_batched_basis_learnable_knots(self):
        """Test batched basis with learnable knots."""
        from torchscience.spline._basis import b_spline_basis

        # Batch of 2 different knot vectors
        knots = torch.stack([
            torch.linspace(0, 1, 8, dtype=torch.float64),
            torch.linspace(0, 2, 8, dtype=torch.float64),
        ])  # (2, 8)

        t = torch.tensor([0.5], dtype=torch.float64)

        # Use vmap to compute basis for each knot vector
        basis = vmap(lambda k: b_spline_basis(k, t, degree=3))(knots)

        assert basis.shape == (2, 1, 4)  # (batch, n_query, n_basis)
```

Run: `uv run python -m pytest tests/torchscience/spline/test__batching.py -v`

```bash
git add tests/torchscience/spline/test__batching.py
git commit -m "test(spline): add comprehensive batching tests"
```

---

## Summary

This plan implements the `torchscience.spline` module in 10 phases:

1. **Module Structure** - Directory layout and exceptions
2. **Tridiagonal Solver** - Thomas algorithm for cubic spline fitting
3. **CubicSpline Fitting** - Tensorclass and fit function with boundary conditions
4. **CubicSpline Evaluation** - Polynomial evaluation with extrapolation modes
5. **CubicSpline Derivatives/Integral** - Analytical derivatives and integration
6. **B-Spline Basis** - Cox-de Boor recursion for basis functions
7. **BSpline Evaluation** - Tensorclass and evaluation via basis functions
8. **BSpline Fitting/Derivatives** - Least squares fitting and derivative formula
9. **Convenience Functions** - Final integration and exports
10. **Batching Tests** - vmap compatibility and batched operations

Each task follows TDD: failing test → implementation → verify → commit.
