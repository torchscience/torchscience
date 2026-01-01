# Phase 1: Dormand-Prince 5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the adaptive Dormand-Prince 5(4) ODE solver with full infrastructure for dense output, TensorDict support, autograd, and CUDA kernels.

**Architecture:** Implement `dormand_prince_5` first as it's the most complex solver. This establishes all patterns: adaptive stepping, dense output via Hermite interpolation, TensorDict flattening/unflattening, differentiable interpolants, and custom CUDA kernels.

**Tech Stack:** PyTorch, TensorDict, C++17 (CPU kernels), CUDA (GPU kernels)

---

## Task 1: Module Structure and Exceptions

**Files:**
- Create: `src/torchscience/integration/__init__.py`
- Create: `src/torchscience/integration/initial_value_problem/__init__.py`
- Create: `src/torchscience/integration/initial_value_problem/_exceptions.py`

**Step 1: Create the integration module init**

```python
# src/torchscience/integration/__init__.py
from torchscience.integration import initial_value_problem

__all__ = [
    "initial_value_problem",
]
```

**Step 2: Create the initial_value_problem submodule init (empty for now)**

```python
# src/torchscience/integration/initial_value_problem/__init__.py
from torchscience.integration.initial_value_problem._exceptions import (
    ConvergenceError,
    MaxStepsExceeded,
    ODESolverError,
    StepSizeError,
)

__all__ = [
    "ConvergenceError",
    "MaxStepsExceeded",
    "ODESolverError",
    "StepSizeError",
]
```

**Step 3: Create the exceptions module**

```python
# src/torchscience/integration/initial_value_problem/_exceptions.py
"""Exceptions for ODE solvers."""


class ODESolverError(Exception):
    """Base exception for ODE solver errors."""

    pass


class MaxStepsExceeded(ODESolverError):
    """Raised when adaptive solver exceeds max_steps."""

    pass


class StepSizeError(ODESolverError):
    """Raised when adaptive step size falls below dt_min."""

    pass


class ConvergenceError(ODESolverError):
    """Raised when implicit solver Newton iteration fails to converge."""

    pass
```

**Step 4: Commit**

```bash
git add src/torchscience/integration/
git commit -m "feat(integration): add module structure and exceptions"
```

---

## Task 2: TensorDict Utilities

**Files:**
- Create: `src/torchscience/integration/initial_value_problem/_tensordict_utils.py`
- Create: `tests/torchscience/integration/initial_value_problem/test__tensordict_utils.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/integration/initial_value_problem/test__tensordict_utils.py
import pytest
import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem._tensordict_utils import (
    flatten_state,
    unflatten_state,
)


class TestFlattenUnflatten:
    def test_tensor_passthrough(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        flat, unflatten = flatten_state(y)
        assert torch.equal(flat, y)
        restored = unflatten(flat)
        assert torch.equal(restored, y)

    def test_simple_tensordict(self):
        y = TensorDict({"x": torch.tensor([1.0, 2.0]), "v": torch.tensor([3.0])})
        flat, unflatten = flatten_state(y)
        assert flat.shape == (3,)
        assert torch.equal(flat, torch.tensor([1.0, 2.0, 3.0]))
        restored = unflatten(flat)
        assert isinstance(restored, TensorDict)
        assert torch.equal(restored["x"], y["x"])
        assert torch.equal(restored["v"], y["v"])

    def test_nested_tensordict(self):
        y = TensorDict(
            {
                "robot": TensorDict(
                    {
                        "joints": torch.tensor([1.0, 2.0]),
                        "velocities": torch.tensor([3.0, 4.0]),
                    }
                ),
                "object": TensorDict({"pose": torch.tensor([5.0, 6.0, 7.0])}),
            }
        )
        flat, unflatten = flatten_state(y)
        assert flat.shape == (7,)
        restored = unflatten(flat)
        assert torch.equal(restored["robot", "joints"], y["robot", "joints"])
        assert torch.equal(restored["object", "pose"], y["object", "pose"])

    def test_batched_tensordict(self):
        y = TensorDict(
            {"x": torch.randn(5, 3), "v": torch.randn(5, 2)}, batch_size=[5]
        )
        flat, unflatten = flatten_state(y)
        assert flat.shape == (5, 5)  # batch_size=5, state_dim=3+2=5
        restored = unflatten(flat)
        assert restored.shape == y.shape
        assert torch.equal(restored["x"], y["x"])

    def test_preserves_gradients(self):
        y = TensorDict(
            {"x": torch.tensor([1.0, 2.0], requires_grad=True)}, batch_size=[]
        )
        flat, unflatten = flatten_state(y)
        assert flat.requires_grad
        restored = unflatten(flat)
        loss = restored["x"].sum()
        loss.backward()
        assert y["x"].grad is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__tensordict_utils.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# src/torchscience/integration/initial_value_problem/_tensordict_utils.py
"""Utilities for flattening and unflattening TensorDict states."""

from typing import Callable, Tuple, Union

import torch
from tensordict import TensorDict


def flatten_state(
    y: Union[torch.Tensor, TensorDict],
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], Union[torch.Tensor, TensorDict]]]:
    """
    Flatten a Tensor or TensorDict to a 1D (or batched 1D) tensor.

    Parameters
    ----------
    y : Tensor or TensorDict
        The state to flatten.

    Returns
    -------
    flat : Tensor
        Flattened state. Shape is (*batch_dims, total_elements).
    unflatten : callable
        Function to restore the original structure from a flat tensor.
    """
    if isinstance(y, torch.Tensor):
        # Tensor passthrough - no flattening needed
        def unflatten_tensor(flat: torch.Tensor) -> torch.Tensor:
            return flat

        return y, unflatten_tensor

    # TensorDict case
    # Get all leaves in consistent order (sorted by key for reproducibility)
    flat_keys = sorted(y.keys(include_nested=True, leaves_only=True))

    # Determine batch dimensions
    batch_size = y.batch_size

    # Collect shapes and flatten each leaf
    shapes = []
    flat_parts = []
    for key in flat_keys:
        leaf = y[key]
        # Shape after batch dimensions
        leaf_shape = leaf.shape[len(batch_size) :]
        shapes.append((key, leaf_shape))
        # Flatten non-batch dimensions
        flat_leaf = leaf.reshape(*batch_size, -1)
        flat_parts.append(flat_leaf)

    # Concatenate along the last dimension
    flat = torch.cat(flat_parts, dim=-1)

    def unflatten_tensordict(flat_tensor: torch.Tensor) -> TensorDict:
        # Split and reshape back
        result = TensorDict({}, batch_size=batch_size)
        offset = 0
        for key, shape in shapes:
            numel = 1
            for s in shape:
                numel *= s
            leaf_flat = flat_tensor[..., offset : offset + numel]
            leaf = leaf_flat.reshape(*batch_size, *shape)
            result[key] = leaf
            offset += numel
        return result

    return flat, unflatten_tensordict


def unflatten_state(
    flat: torch.Tensor,
    template: Union[torch.Tensor, TensorDict],
) -> Union[torch.Tensor, TensorDict]:
    """
    Unflatten a flat tensor using a template for structure.

    Parameters
    ----------
    flat : Tensor
        Flattened state tensor.
    template : Tensor or TensorDict
        Template providing the target structure.

    Returns
    -------
    y : Tensor or TensorDict
        Unflattened state matching template structure.
    """
    _, unflatten_fn = flatten_state(template)
    return unflatten_fn(flat)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__tensordict_utils.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/_tensordict_utils.py tests/torchscience/integration/initial_value_problem/
git commit -m "feat(ivp): add TensorDict flatten/unflatten utilities"
```

---

## Task 3: Interpolant Base Class

**Files:**
- Create: `src/torchscience/integration/initial_value_problem/_interpolant.py`
- Create: `tests/torchscience/integration/initial_value_problem/test__interpolant.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/integration/initial_value_problem/test__interpolant.py
import pytest
import torch

from torchscience.integration.initial_value_problem._interpolant import (
    HermiteInterpolant,
)


class TestHermiteInterpolant:
    def test_basic_interpolation(self):
        # Linear function y = t, so y' = 1
        t_points = torch.tensor([0.0, 1.0, 2.0])
        y_points = torch.tensor([[0.0], [1.0], [2.0]])
        dy_points = torch.tensor([[1.0], [1.0], [1.0]])

        interp = HermiteInterpolant(t_points, y_points, dy_points)

        # Query at midpoints
        y_half = interp(0.5)
        assert torch.allclose(y_half, torch.tensor([0.5]), atol=1e-6)

        y_1_5 = interp(1.5)
        assert torch.allclose(y_1_5, torch.tensor([1.5]), atol=1e-6)

    def test_quadratic_interpolation(self):
        # y = t^2, y' = 2t
        t_points = torch.tensor([0.0, 1.0, 2.0])
        y_points = torch.tensor([[0.0], [1.0], [4.0]])
        dy_points = torch.tensor([[0.0], [2.0], [4.0]])

        interp = HermiteInterpolant(t_points, y_points, dy_points)

        # Hermite should exactly interpolate polynomials up to degree 3
        y_half = interp(0.5)
        assert torch.allclose(y_half, torch.tensor([0.25]), atol=1e-5)

    def test_multiple_queries(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [1.0]])
        dy_points = torch.tensor([[1.0], [1.0]])

        interp = HermiteInterpolant(t_points, y_points, dy_points)

        t_query = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        y_query = interp(t_query)
        assert y_query.shape == (5, 1)
        expected = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
        assert torch.allclose(y_query, expected, atol=1e-6)

    def test_out_of_bounds_raises(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [1.0]])
        dy_points = torch.tensor([[1.0], [1.0]])

        interp = HermiteInterpolant(t_points, y_points, dy_points)

        with pytest.raises(ValueError, match="outside"):
            interp(-0.1)

        with pytest.raises(ValueError, match="outside"):
            interp(1.1)

    def test_differentiable(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [1.0]], requires_grad=True)
        dy_points = torch.tensor([[1.0], [1.0]])

        interp = HermiteInterpolant(t_points, y_points, dy_points)

        y_mid = interp(0.5)
        loss = y_mid.sum()
        loss.backward()

        assert y_points.grad is not None

    def test_success_attribute(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [1.0]])
        dy_points = torch.tensor([[1.0], [1.0]])

        interp = HermiteInterpolant(t_points, y_points, dy_points)
        assert interp.success is None  # default

        success = torch.tensor([True, False])
        interp = HermiteInterpolant(t_points, y_points, dy_points, success=success)
        assert torch.equal(interp.success, success)

    def test_batched_state(self):
        # Batch of 3 trajectories, state dim 2
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.randn(2, 3, 2)  # (T, B, D)
        dy_points = torch.randn(2, 3, 2)

        interp = HermiteInterpolant(t_points, y_points, dy_points)

        # Query single time
        y_mid = interp(0.5)
        assert y_mid.shape == (3, 2)

        # Query multiple times
        t_query = torch.tensor([0.25, 0.5, 0.75])
        y_query = interp(t_query)
        assert y_query.shape == (3, 3, 2)  # (T_query, B, D)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__interpolant.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/torchscience/integration/initial_value_problem/_interpolant.py
"""Interpolants for dense output from ODE solvers."""

from typing import Optional, Union

import torch


class HermiteInterpolant:
    """
    Cubic Hermite interpolant for ODE dense output.

    Uses function values and derivatives at grid points to construct
    a C1-continuous interpolant. The interpolant is differentiable
    and supports backpropagation.

    Parameters
    ----------
    t_points : Tensor
        Time points, shape (N,), must be monotonically increasing.
    y_points : Tensor
        State values at time points, shape (N, *state_shape).
    dy_points : Tensor
        Derivative values at time points, shape (N, *state_shape).
    success : Tensor, optional
        Boolean mask indicating which batch elements succeeded.
        Shape (*batch_shape,). Only set when throw=False.
    """

    def __init__(
        self,
        t_points: torch.Tensor,
        y_points: torch.Tensor,
        dy_points: torch.Tensor,
        success: Optional[torch.Tensor] = None,
    ):
        self.t_points = t_points
        self.y_points = y_points
        self.dy_points = dy_points
        self.success = success
        self._t_min = t_points[0].item()
        self._t_max = t_points[-1].item()

    def __call__(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the interpolant at time(s) t.

        Parameters
        ----------
        t : float or Tensor
            Time(s) to query. Scalar or 1D tensor.

        Returns
        -------
        y : Tensor
            State at time(s) t.
            If t is scalar: shape (*state_shape) or (*batch_shape, *state_shape)
            If t is 1D tensor of length T: shape (T, *batch_shape, *state_shape)
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=self.t_points.dtype, device=self.t_points.device)

        scalar_query = t.dim() == 0
        if scalar_query:
            t = t.unsqueeze(0)

        # Validate bounds
        t_min_query = t.min().item()
        t_max_query = t.max().item()
        if t_min_query < self._t_min - 1e-6 or t_max_query > self._t_max + 1e-6:
            raise ValueError(
                f"Query time(s) outside interpolant range [{self._t_min}, {self._t_max}]"
            )

        # Find interval for each query point
        # searchsorted returns index where t would be inserted
        indices = torch.searchsorted(self.t_points, t.contiguous())
        indices = indices.clamp(1, len(self.t_points) - 1)

        # Get interval endpoints
        t0 = self.t_points[indices - 1]
        t1 = self.t_points[indices]
        y0 = self.y_points[indices - 1]
        y1 = self.y_points[indices]
        dy0 = self.dy_points[indices - 1]
        dy1 = self.dy_points[indices]

        # Normalized position in interval
        h = t1 - t0
        s = (t - t0) / h

        # Expand s for broadcasting with state dimensions
        # s has shape (T_query,), need to broadcast with (*state_shape)
        state_dims = y0.dim() - 1  # exclude the time dimension (first)
        for _ in range(state_dims):
            s = s.unsqueeze(-1)
            h = h.unsqueeze(-1)

        # Cubic Hermite basis functions
        # H00(s) = 2s^3 - 3s^2 + 1
        # H10(s) = s^3 - 2s^2 + s
        # H01(s) = -2s^3 + 3s^2
        # H11(s) = s^3 - s^2
        s2 = s * s
        s3 = s2 * s

        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2

        # Interpolate
        y = h00 * y0 + h10 * h * dy0 + h01 * y1 + h11 * h * dy1

        if scalar_query:
            y = y.squeeze(0)

        return y
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__interpolant.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/_interpolant.py tests/torchscience/integration/initial_value_problem/test__interpolant.py
git commit -m "feat(ivp): add Hermite interpolant for dense output"
```

---

## Task 4: Dormand-Prince 5 Core Implementation (Python)

**Files:**
- Create: `src/torchscience/integration/initial_value_problem/_dormand_prince_5.py`
- Create: `tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py`

**Step 1: Write the failing test for basic functionality**

```python
# tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py
import pytest
import torch

from torchscience.integration.initial_value_problem import dormand_prince_5


class TestDormandPrince5Basic:
    def test_exponential_decay(self):
        """Test against analytical solution: dy/dt = -y, y(0) = 1 => y(t) = exp(-t)"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(decay, y0, t_span=(0.0, 5.0))

        expected = torch.exp(torch.tensor([-5.0]))
        assert torch.allclose(y_final, expected, rtol=1e-4)

    def test_harmonic_oscillator(self):
        """Test 2D system: simple harmonic oscillator"""

        def oscillator(t, y):
            x, v = y[..., 0], y[..., 1]
            return torch.stack(
                [v, -x], dim=-1
            )  # dx/dt = v, dv/dt = -x

        y0 = torch.tensor([1.0, 0.0])  # x=1, v=0
        y_final, interp = dormand_prince_5(oscillator, y0, t_span=(0.0, 2 * torch.pi))

        # After one period, should return to initial state
        assert torch.allclose(y_final, y0, atol=1e-3)

    def test_returns_interpolant(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        # Interpolant should be callable
        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_interpolant_endpoints(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        # At t=0, should match y0
        assert torch.allclose(interp(0.0), y0, atol=1e-6)
        # At t=1, should match y_final
        assert torch.allclose(interp(1.0), y_final, atol=1e-6)

    def test_multiple_time_queries(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        t_query = torch.linspace(0, 1, 10)
        trajectory = interp(t_query)
        assert trajectory.shape == (10, 1)

        # Should be monotonically decreasing
        for i in range(9):
            assert trajectory[i, 0] > trajectory[i + 1, 0]


class TestDormandPrince5Tolerances:
    def test_tighter_tolerance_improves_accuracy(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        expected = torch.exp(torch.tensor([-1.0]))

        y_loose, _ = dormand_prince_5(f, y0, t_span=(0.0, 1.0), rtol=1e-3, atol=1e-6)
        y_tight, _ = dormand_prince_5(f, y0, t_span=(0.0, 1.0), rtol=1e-7, atol=1e-10)

        error_loose = (y_loose - expected).abs().item()
        error_tight = (y_tight - expected).abs().item()

        assert error_tight < error_loose


class TestDormandPrince5Autograd:
    def test_gradient_through_solver(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        y_final, _ = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad).any()

    def test_gradient_through_interpolant(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        _, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        y_mid = interp(0.5)
        loss = y_mid.sum()
        loss.backward()

        assert theta.grad is not None


class TestDormandPrince5Batched:
    def test_batched_initial_conditions(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        assert y_final.shape == (3, 1)
        expected = y0 * torch.exp(torch.tensor([-1.0]))
        assert torch.allclose(y_final, expected, rtol=1e-4)

    def test_batched_interpolant(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0], [2.0]])  # (2, 1)
        _, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        y_mid = interp(0.5)
        assert y_mid.shape == (2, 1)

        t_query = torch.tensor([0.25, 0.5, 0.75])
        trajectory = interp(t_query)
        assert trajectory.shape == (3, 2, 1)  # (T, B, D)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py::TestDormandPrince5Basic -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/torchscience/integration/initial_value_problem/_dormand_prince_5.py
"""Dormand-Prince 5(4) adaptive ODE solver."""

from typing import Callable, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem._exceptions import (
    MaxStepsExceeded,
    StepSizeError,
)
from torchscience.integration.initial_value_problem._interpolant import (
    HermiteInterpolant,
)
from torchscience.integration.initial_value_problem._tensordict_utils import (
    flatten_state,
)

# Dormand-Prince 5(4) Butcher tableau coefficients
# fmt: off
_C = torch.tensor([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])
_A = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
]
_B5 = torch.tensor([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
_B4 = torch.tensor([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
# fmt: on


def dormand_prince_5(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    dt0: Optional[float] = None,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    max_steps: int = 10000,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using Dormand-Prince 5(4) adaptive method.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
        Use closures or functools.partial to pass additional parameters.
    y0 : Tensor or TensorDict
        Initial state
    t_span : tuple[float, float]
        Integration interval (t0, t1). Supports backward integration if t1 < t0.
    rtol : float
        Relative tolerance for step size control
    atol : float
        Absolute tolerance for step size control
    dt0 : float, optional
        Initial step size guess. If None, estimated automatically.
    dt_min : float, optional
        Minimum allowed step size. Raises StepSizeError if step would go below
        (when throw=True).
    dt_max : float, optional
        Maximum allowed step size. Step size is clamped to this value.
    max_steps : int
        Maximum number of steps before raising MaxStepsExceeded (when throw=True).
    throw : bool
        If True (default), raise exceptions on solver failures. If False, return
        NaN for failed batch elements and attach `success` mask to interpolant.

    Returns
    -------
    y : Tensor or TensorDict
        State at t1, shape (*state_shape). Differentiable.
    interp : callable
        Interpolant function. interp(t) returns state at time(s) t.
        Differentiable. Has `success` attribute (bool Tensor) when throw=False.

    Raises
    ------
    MaxStepsExceeded
        If integration requires more than max_steps (only when throw=True).
    StepSizeError
        If adaptive step size falls below dt_min (only when throw=True).
    """
    t0, t1 = t_span
    direction = 1.0 if t1 >= t0 else -1.0
    t_end = t1

    # Handle TensorDict
    is_tensordict = isinstance(y0, TensorDict)
    y_flat, unflatten = flatten_state(y0)

    # Wrap dynamics for flattened state
    if is_tensordict:

        def f_flat(t, y):
            y_struct = unflatten(y)
            dy_struct = f(t, y_struct)
            dy_flat, _ = flatten_state(dy_struct)
            return dy_flat

    else:
        f_flat = f

    # Get dtype and device
    dtype = y_flat.dtype
    device = y_flat.device

    # Move Butcher tableau to correct dtype/device
    C = _C.to(dtype=dtype, device=device)
    A = [[torch.tensor(a, dtype=dtype, device=device) for a in row] for row in _A]
    B5 = _B5.to(dtype=dtype, device=device)
    B4 = _B4.to(dtype=dtype, device=device)

    # Estimate initial step size if not provided
    if dt0 is None:
        f0 = f_flat(t0, y_flat)
        scale = atol + rtol * torch.abs(y_flat)
        d0 = torch.sqrt(torch.mean((y_flat / scale) ** 2))
        d1 = torch.sqrt(torch.mean((f0 / scale) ** 2))
        if d0 < 1e-5 or d1 < 1e-5:
            dt0 = 1e-6
        else:
            dt0 = 0.01 * (d0 / d1).item()
    dt = dt0

    # Apply dt_max
    if dt_max is not None:
        dt = min(dt, dt_max)

    # Storage for interpolant
    t_points = [torch.tensor(t0, dtype=dtype, device=device)]
    y_points = [y_flat.clone()]
    dy_points = [f_flat(t0, y_flat)]

    t = t0
    y = y_flat.clone()
    n_steps = 0
    success = None if throw else torch.ones(y_flat.shape[:-1] if y_flat.dim() > 1 else (), dtype=torch.bool, device=device)

    while direction * (t_end - t) > 1e-10:
        if n_steps >= max_steps:
            if throw:
                raise MaxStepsExceeded(
                    f"Exceeded maximum number of steps ({max_steps})"
                )
            else:
                y = torch.full_like(y, float("nan"))
                if success is not None:
                    success = torch.zeros_like(success)
                break

        # Clamp step to not overshoot
        dt_step = min(dt, abs(t_end - t))
        h = direction * dt_step

        # Compute RK stages
        k = [None] * 7
        k[0] = f_flat(t, y)
        for i in range(1, 7):
            t_i = t + C[i] * h
            y_i = y.clone()
            for j, a_ij in enumerate(A[i]):
                y_i = y_i + h * a_ij * k[j]
            k[i] = f_flat(t_i, y_i)

        # 5th order solution
        y_new = y.clone()
        for i, b in enumerate(B5):
            y_new = y_new + h * b * k[i]

        # 4th order solution for error estimate
        y_err = y.clone()
        for i, b in enumerate(B4):
            y_err = y_err + h * b * k[i]

        # Error estimate
        error = y_new - y_err
        scale = atol + rtol * torch.maximum(torch.abs(y), torch.abs(y_new))
        err_norm = torch.sqrt(torch.mean((error / scale) ** 2))

        # Step accepted?
        if err_norm <= 1.0:
            t = t + h
            y = y_new

            # Store for interpolant
            t_points.append(torch.tensor(t, dtype=dtype, device=device))
            y_points.append(y.clone())
            dy_points.append(k[6].clone())  # FSAL: k[6] = f(t_new, y_new)

            n_steps += 1

        # Adjust step size
        if err_norm == 0:
            factor = 2.0
        else:
            factor = 0.9 * (1.0 / err_norm) ** 0.2
        factor = max(0.1, min(factor, 5.0))  # Limit growth/shrinkage
        dt = dt_step * factor

        if dt_max is not None:
            dt = min(dt, dt_max)
        if dt_min is not None and dt < dt_min:
            if throw:
                raise StepSizeError(f"Step size {dt} below minimum {dt_min}")
            else:
                y = torch.full_like(y, float("nan"))
                if success is not None:
                    success = torch.zeros_like(success)
                break

    # Build interpolant
    t_tensor = torch.stack(t_points)
    y_tensor = torch.stack(y_points)
    dy_tensor = torch.stack(dy_points)

    interp = HermiteInterpolant(t_tensor, y_tensor, dy_tensor, success=success)

    # Wrap interpolant for TensorDict
    if is_tensordict:

        def interp_tensordict(t_query):
            y_flat_query = interp(t_query)
            if isinstance(t_query, (int, float)) or (
                isinstance(t_query, torch.Tensor) and t_query.dim() == 0
            ):
                return unflatten(y_flat_query)
            else:
                # Multiple times: unflatten each time point
                return torch.stack(
                    [unflatten(y_flat_query[i]) for i in range(y_flat_query.shape[0])]
                )

        interp_tensordict.success = interp.success
        final_interp = interp_tensordict
    else:
        final_interp = interp

    # Unflatten final result
    if is_tensordict:
        y_final = unflatten(y)
    else:
        y_final = y

    return y_final, final_interp
```

**Step 4: Update the module init**

```python
# src/torchscience/integration/initial_value_problem/__init__.py
from torchscience.integration.initial_value_problem._dormand_prince_5 import (
    dormand_prince_5,
)
from torchscience.integration.initial_value_problem._exceptions import (
    ConvergenceError,
    MaxStepsExceeded,
    ODESolverError,
    StepSizeError,
)

__all__ = [
    "ConvergenceError",
    "MaxStepsExceeded",
    "ODESolverError",
    "StepSizeError",
    "dormand_prince_5",
]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/ tests/torchscience/integration/initial_value_problem/
git commit -m "feat(ivp): add dormand_prince_5 adaptive ODE solver"
```

---

## Task 5: Complex Number Support Tests

**Files:**
- Modify: `tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py`

**Step 1: Write failing tests for complex support**

Add to the test file:

```python
class TestDormandPrince5Complex:
    def test_complex_exponential_decay(self):
        """Test complex ODE: dy/dt = -i*y, y(0) = 1 => y(t) = exp(-i*t)"""

        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, torch.pi))

        expected = torch.exp(-1j * torch.tensor(torch.pi))
        assert torch.allclose(y_final, expected.unsqueeze(0), atol=1e-4)

    def test_schrodinger_like(self):
        """Test Schrodinger-like equation: dy/dt = -i*H*y"""
        H = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)

        def f(t, y):
            return -1j * H @ y

        psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
        y_final, _ = dormand_prince_5(f, psi0, t_span=(0.0, 1.0))

        # Check normalization preserved
        norm = torch.abs(y_final).pow(2).sum()
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_complex_gradcheck(self):
        """Verify gradients work for complex states"""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta.to(y.dtype) * 1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        # Can compute gradients through complex operations
        loss = y_final.abs().sum()
        loss.backward()

        assert theta.grad is not None
```

**Step 2: Run test to verify it passes (implementation should already support this)**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py::TestDormandPrince5Complex -v`
Expected: PASS (if not, fix implementation)

**Step 3: Commit**

```bash
git add tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py
git commit -m "test(ivp): add complex number support tests for dormand_prince_5"
```

---

## Task 6: TensorDict Integration Tests

**Files:**
- Modify: `tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py`

**Step 1: Write failing tests for TensorDict support**

Add to the test file:

```python
from tensordict import TensorDict


class TestDormandPrince5TensorDict:
    def test_simple_tensordict(self):
        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict({"x": torch.tensor([1.0]), "v": torch.tensor([0.0])})
        state_final, interp = dormand_prince_5(f, state0, t_span=(0.0, 2 * torch.pi))

        # After one period, should return to initial state
        assert isinstance(state_final, TensorDict)
        assert torch.allclose(state_final["x"], state0["x"], atol=1e-3)
        assert torch.allclose(state_final["v"], state0["v"], atol=1e-3)

    def test_nested_tensordict(self):
        def f(t, state):
            return TensorDict(
                {
                    "robot": TensorDict(
                        {
                            "q": state["robot"]["dq"],
                            "dq": -state["robot"]["q"],
                        }
                    )
                }
            )

        state0 = TensorDict(
            {
                "robot": TensorDict(
                    {
                        "q": torch.tensor([1.0, 0.0]),
                        "dq": torch.tensor([0.0, 1.0]),
                    }
                )
            }
        )

        state_final, interp = dormand_prince_5(f, state0, t_span=(0.0, 1.0))

        assert isinstance(state_final, TensorDict)
        assert "robot" in state_final.keys()
        assert state_final["robot", "q"].shape == (2,)

    def test_tensordict_interpolant(self):
        def f(t, state):
            return TensorDict({"x": -state["x"]})

        state0 = TensorDict({"x": torch.tensor([1.0])})
        _, interp = dormand_prince_5(f, state0, t_span=(0.0, 1.0))

        state_mid = interp(0.5)
        assert isinstance(state_mid, TensorDict)
        assert "x" in state_mid.keys()
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py::TestDormandPrince5TensorDict -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py
git commit -m "test(ivp): add TensorDict support tests for dormand_prince_5"
```

---

## Task 7: Error Handling Tests

**Files:**
- Modify: `tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py`

**Step 1: Write tests for error handling**

Add to the test file:

```python
class TestDormandPrince5ErrorHandling:
    def test_max_steps_exceeded_throws(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])

        with pytest.raises(MaxStepsExceeded):
            dormand_prince_5(f, y0, t_span=(0.0, 1000.0), max_steps=5)

    def test_max_steps_exceeded_no_throw(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])

        y_final, interp = dormand_prince_5(
            f, y0, t_span=(0.0, 1000.0), max_steps=5, throw=False
        )

        assert torch.isnan(y_final).all()
        assert interp.success is not None
        assert not interp.success.all()

    def test_step_size_error_throws(self):
        # Stiff problem that requires tiny steps
        def stiff(t, y):
            return -1000 * y

        y0 = torch.tensor([1.0])

        with pytest.raises(StepSizeError):
            dormand_prince_5(f, y0, t_span=(0.0, 1.0), dt_min=0.1)

    def test_interpolant_out_of_bounds(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        _, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        with pytest.raises(ValueError, match="outside"):
            interp(-0.1)

        with pytest.raises(ValueError, match="outside"):
            interp(1.1)


from torchscience.integration.initial_value_problem import (
    MaxStepsExceeded,
    StepSizeError,
)
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py::TestDormandPrince5ErrorHandling -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py
git commit -m "test(ivp): add error handling tests for dormand_prince_5"
```

---

## Task 8: Backward Integration Tests

**Files:**
- Modify: `tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py`

**Step 1: Write tests for backward integration**

Add to the test file:

```python
class TestDormandPrince5BackwardIntegration:
    def test_backward_exponential(self):
        """Integrate backwards: y(1) = e^-1 => y(0) = 1"""

        def f(t, y):
            return -y

        y1 = torch.tensor([torch.exp(torch.tensor(-1.0))])
        y0_recovered, interp = dormand_prince_5(f, y1, t_span=(1.0, 0.0))

        expected = torch.tensor([1.0])
        assert torch.allclose(y0_recovered, expected, rtol=1e-4)

    def test_backward_interpolant_range(self):
        """Interpolant should cover [0, 1] regardless of direction"""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        _, interp = dormand_prince_5(f, y0, t_span=(1.0, 0.0))

        # Should be able to query anywhere in [0, 1]
        y_mid = interp(0.5)
        assert not torch.isnan(y_mid).any()
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py::TestDormandPrince5BackwardIntegration -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py
git commit -m "test(ivp): add backward integration tests for dormand_prince_5"
```

---

## Task 9: CPU C++ Kernel (Header Only)

**Files:**
- Create: `src/torchscience/csrc/cpu/integration/initial_value_problem/dormand_prince_5.h`

**Step 1: Write the C++ kernel header**

```cpp
// src/torchscience/csrc/cpu/integration/initial_value_problem/dormand_prince_5.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/util/Exception.h>
#include <cmath>
#include <tuple>
#include <vector>

namespace torchscience {
namespace cpu {
namespace integration {
namespace initial_value_problem {

// Dormand-Prince 5(4) Butcher tableau coefficients
template <typename scalar_t>
struct DormandPrince5Tableau {
  static constexpr scalar_t c2 = 1.0 / 5.0;
  static constexpr scalar_t c3 = 3.0 / 10.0;
  static constexpr scalar_t c4 = 4.0 / 5.0;
  static constexpr scalar_t c5 = 8.0 / 9.0;
  static constexpr scalar_t c6 = 1.0;
  static constexpr scalar_t c7 = 1.0;

  static constexpr scalar_t a21 = 1.0 / 5.0;
  static constexpr scalar_t a31 = 3.0 / 40.0;
  static constexpr scalar_t a32 = 9.0 / 40.0;
  static constexpr scalar_t a41 = 44.0 / 45.0;
  static constexpr scalar_t a42 = -56.0 / 15.0;
  static constexpr scalar_t a43 = 32.0 / 9.0;
  static constexpr scalar_t a51 = 19372.0 / 6561.0;
  static constexpr scalar_t a52 = -25360.0 / 2187.0;
  static constexpr scalar_t a53 = 64448.0 / 6561.0;
  static constexpr scalar_t a54 = -212.0 / 729.0;
  static constexpr scalar_t a61 = 9017.0 / 3168.0;
  static constexpr scalar_t a62 = -355.0 / 33.0;
  static constexpr scalar_t a63 = 46732.0 / 5247.0;
  static constexpr scalar_t a64 = 49.0 / 176.0;
  static constexpr scalar_t a65 = -5103.0 / 18656.0;
  static constexpr scalar_t a71 = 35.0 / 384.0;
  static constexpr scalar_t a72 = 0.0;
  static constexpr scalar_t a73 = 500.0 / 1113.0;
  static constexpr scalar_t a74 = 125.0 / 192.0;
  static constexpr scalar_t a75 = -2187.0 / 6784.0;
  static constexpr scalar_t a76 = 11.0 / 84.0;

  // 5th order weights (same as a7*)
  static constexpr scalar_t b51 = 35.0 / 384.0;
  static constexpr scalar_t b52 = 0.0;
  static constexpr scalar_t b53 = 500.0 / 1113.0;
  static constexpr scalar_t b54 = 125.0 / 192.0;
  static constexpr scalar_t b55 = -2187.0 / 6784.0;
  static constexpr scalar_t b56 = 11.0 / 84.0;
  static constexpr scalar_t b57 = 0.0;

  // 4th order weights (for error estimation)
  static constexpr scalar_t b41 = 5179.0 / 57600.0;
  static constexpr scalar_t b42 = 0.0;
  static constexpr scalar_t b43 = 7571.0 / 16695.0;
  static constexpr scalar_t b44 = 393.0 / 640.0;
  static constexpr scalar_t b45 = -92097.0 / 339200.0;
  static constexpr scalar_t b46 = 187.0 / 2100.0;
  static constexpr scalar_t b47 = 1.0 / 40.0;
};

template <typename scalar_t>
struct DormandPrince5Step {
  // Single step of Dormand-Prince 5(4) method
  // Returns: (y_new, error_estimate, k7) where k7 can be reused as k1 for next step (FSAL)
  static std::tuple<at::Tensor, at::Tensor, at::Tensor> step(
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& f,
      scalar_t t,
      const at::Tensor& y,
      scalar_t h,
      const at::Tensor& k1  // Can reuse from previous step (FSAL)
  ) {
    using T = DormandPrince5Tableau<scalar_t>;

    at::Tensor k2 = f(t + T::c2 * h, y + h * T::a21 * k1);
    at::Tensor k3 = f(t + T::c3 * h, y + h * (T::a31 * k1 + T::a32 * k2));
    at::Tensor k4 = f(t + T::c4 * h, y + h * (T::a41 * k1 + T::a42 * k2 + T::a43 * k3));
    at::Tensor k5 = f(t + T::c5 * h, y + h * (T::a51 * k1 + T::a52 * k2 + T::a53 * k3 + T::a54 * k4));
    at::Tensor k6 = f(t + T::c6 * h, y + h * (T::a61 * k1 + T::a62 * k2 + T::a63 * k3 + T::a64 * k4 + T::a65 * k5));

    // 5th order solution
    at::Tensor y_new = y + h * (T::b51 * k1 + T::b52 * k2 + T::b53 * k3 + T::b54 * k4 + T::b55 * k5 + T::b56 * k6);

    // k7 for FSAL property (equals f(t+h, y_new))
    at::Tensor k7 = f(t + h, y_new);

    // Error estimate (difference between 5th and 4th order solutions)
    at::Tensor error = h * (
        (T::b51 - T::b41) * k1 +
        (T::b52 - T::b42) * k2 +
        (T::b53 - T::b43) * k3 +
        (T::b54 - T::b44) * k4 +
        (T::b55 - T::b45) * k5 +
        (T::b56 - T::b46) * k6 +
        (T::b57 - T::b47) * k7
    );

    return std::make_tuple(y_new, error, k7);
  }
};

// Error norm computation for step size control
template <typename scalar_t>
scalar_t compute_error_norm(
    const at::Tensor& error,
    const at::Tensor& y,
    const at::Tensor& y_new,
    scalar_t atol,
    scalar_t rtol
) {
  at::Tensor scale = atol + rtol * at::maximum(at::abs(y), at::abs(y_new));
  at::Tensor normalized_error = error / scale;
  scalar_t err_norm = std::sqrt(at::mean(normalized_error * normalized_error).item<scalar_t>());
  return err_norm;
}

// Step size adjustment
template <typename scalar_t>
scalar_t adjust_step_size(
    scalar_t dt,
    scalar_t err_norm,
    scalar_t dt_min,
    scalar_t dt_max,
    scalar_t safety = 0.9,
    scalar_t min_factor = 0.1,
    scalar_t max_factor = 5.0
) {
  scalar_t factor;
  if (err_norm == 0) {
    factor = max_factor;
  } else {
    // PI controller: (1/err)^(1/5) for 5th order method
    factor = safety * std::pow(1.0 / err_norm, 0.2);
  }
  factor = std::max(min_factor, std::min(factor, max_factor));

  scalar_t dt_new = dt * factor;
  if (dt_max > 0) {
    dt_new = std::min(dt_new, dt_max);
  }
  if (dt_min > 0) {
    dt_new = std::max(dt_new, dt_min);
  }
  return dt_new;
}

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cpu
}  // namespace torchscience
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/integration/initial_value_problem/
git commit -m "feat(ivp): add Dormand-Prince 5 CPU kernel header"
```

---

## Task 10: CUDA Kernel (Placeholder)

**Files:**
- Create: `src/torchscience/csrc/cuda/integration/initial_value_problem/dormand_prince_5.cu`

**Step 1: Write the CUDA kernel placeholder**

```cpp
// src/torchscience/csrc/cuda/integration/initial_value_problem/dormand_prince_5.cu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace torchscience {
namespace cuda {
namespace integration {
namespace initial_value_problem {

// TODO: Implement CUDA kernel for Dormand-Prince 5(4)
//
// Key optimizations to consider:
// 1. Fused kernel for computing all 7 RK stages
// 2. Shared memory for intermediate k values
// 3. Warp-level reduction for error norm computation
// 4. Batched integration (different initial conditions in parallel)
//
// For now, the Python implementation dispatches to CPU for CUDA tensors.
// This placeholder is for future CUDA-native implementation.

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cuda
}  // namespace torchscience
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cuda/integration/initial_value_problem/
git commit -m "feat(ivp): add Dormand-Prince 5 CUDA kernel placeholder"
```

---

## Task 11: Integration Tests with SciPy Reference

**Files:**
- Modify: `tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py`

**Step 1: Add SciPy comparison tests**

Add to the test file:

```python
scipy = pytest.importorskip("scipy")
from scipy.integrate import solve_ivp


class TestDormandPrince5SciPyComparison:
    def test_exponential_decay_matches_scipy(self):
        def f_torch(t, y):
            return -y

        def f_scipy(t, y):
            return -y

        y0_val = 1.0
        t_span = (0.0, 5.0)

        # Solve with torchscience
        y0_torch = torch.tensor([y0_val], dtype=torch.float64)
        y_torch, _ = dormand_prince_5(f_torch, y0_torch, t_span, rtol=1e-8, atol=1e-10)

        # Solve with scipy
        sol_scipy = solve_ivp(f_scipy, t_span, [y0_val], method="DOP853", rtol=1e-8, atol=1e-10)

        assert torch.allclose(
            y_torch, torch.tensor(sol_scipy.y[:, -1], dtype=torch.float64), rtol=1e-5
        )

    def test_lotka_volterra_matches_scipy(self):
        """Lotka-Volterra predator-prey model"""
        alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0

        def f_torch(t, y):
            x, p = y[..., 0], y[..., 1]
            dx = alpha * x - beta * x * p
            dp = delta * x * p - gamma * p
            return torch.stack([dx, dp], dim=-1)

        def f_scipy(t, y):
            x, p = y
            dx = alpha * x - beta * x * p
            dp = delta * x * p - gamma * p
            return [dx, dp]

        y0_val = [10.0, 5.0]
        t_span = (0.0, 10.0)

        y0_torch = torch.tensor(y0_val, dtype=torch.float64)
        y_torch, _ = dormand_prince_5(f_torch, y0_torch, t_span, rtol=1e-8, atol=1e-10)

        sol_scipy = solve_ivp(f_scipy, t_span, y0_val, method="DOP853", rtol=1e-8, atol=1e-10)

        assert torch.allclose(
            y_torch, torch.tensor(sol_scipy.y[:, -1], dtype=torch.float64), rtol=1e-4
        )
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py::TestDormandPrince5SciPyComparison -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/torchscience/integration/initial_value_problem/test__dormand_prince_5.py
git commit -m "test(ivp): add SciPy comparison tests for dormand_prince_5"
```

---

## Task 12: Final Documentation and Export

**Files:**
- Modify: `src/torchscience/integration/initial_value_problem/__init__.py`
- Modify: `src/torchscience/integration/__init__.py`

**Step 1: Update exports with docstrings**

```python
# src/torchscience/integration/initial_value_problem/__init__.py
"""
Initial value problem solvers for ordinary differential equations.

This module provides differentiable ODE solvers for PyTorch tensors and TensorDict.

Available Solvers
-----------------
dormand_prince_5
    Adaptive 5th-order Runge-Kutta method (Dormand-Prince 5(4)).
    Production-quality solver for most non-stiff problems.

Examples
--------
>>> import torch
>>> from torchscience.integration.initial_value_problem import dormand_prince_5
>>>
>>> def decay(t, y):
...     return -y
>>>
>>> y0 = torch.tensor([1.0])
>>> y_final, interp = dormand_prince_5(decay, y0, t_span=(0.0, 5.0))
>>> trajectory = interp(torch.linspace(0, 5, 100))
"""

from torchscience.integration.initial_value_problem._dormand_prince_5 import (
    dormand_prince_5,
)
from torchscience.integration.initial_value_problem._exceptions import (
    ConvergenceError,
    MaxStepsExceeded,
    ODESolverError,
    StepSizeError,
)

__all__ = [
    "ConvergenceError",
    "MaxStepsExceeded",
    "ODESolverError",
    "StepSizeError",
    "dormand_prince_5",
]
```

**Step 2: Commit**

```bash
git add src/torchscience/integration/
git commit -m "docs(ivp): add module docstrings and complete Phase 1"
```

---

## Summary

Phase 1 implements the core `dormand_prince_5` adaptive ODE solver with:

1. **Module structure** - `torchscience.integration.initial_value_problem`
2. **Exception classes** - `ODESolverError`, `MaxStepsExceeded`, `StepSizeError`, `ConvergenceError`
3. **TensorDict utilities** - Flatten/unflatten for structured state
4. **Hermite interpolant** - Differentiable dense output
5. **dormand_prince_5 solver** - Full adaptive solver with:
   - Adaptive step size control
   - Dense output via Hermite interpolation
   - TensorDict support
   - Complex number support
   - Autograd support
   - Backward integration
   - Error handling (throw/no-throw modes)
6. **CPU C++ kernel** - Header-only implementation
7. **CUDA kernel placeholder** - For future GPU acceleration
8. **Comprehensive tests** - Including SciPy comparison

This establishes all patterns for Phase 2 (fixed-step solvers), Phase 3 (implicit solver), and Phase 4 (adjoint wrapper).
