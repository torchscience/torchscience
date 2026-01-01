# Phase 2: Fixed-Step Explicit Solvers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three fixed-step explicit ODE solvers (euler, midpoint, runge_kutta_4) using the infrastructure established in Phase 1.

**Architecture:** Each solver follows the same pattern: validate inputs, flatten TensorDict if needed, step through time with fixed dt, build linear interpolant from stored points, return (y_final, interp). Reuse `_tensordict_utils.py` and `_interpolant.py` from Phase 1.

**Tech Stack:** PyTorch, TensorDict, C++17 (CPU kernels), CUDA (GPU kernels)

**Prerequisites:** Phase 1 complete (dormand_prince_5, TensorDict utilities, interpolant, exceptions)

---

## Task 1: Linear Interpolant

**Files:**
- Modify: `src/torchscience/integration/initial_value_problem/_interpolant.py`
- Modify: `tests/torchscience/integration/initial_value_problem/test__interpolant.py`

**Step 1: Write the failing test**

Add to `tests/torchscience/integration/initial_value_problem/test__interpolant.py`:

```python
from torchscience.integration.initial_value_problem._interpolant import (
    LinearInterpolant,
)


class TestLinearInterpolant:
    def test_basic_interpolation(self):
        t_points = torch.tensor([0.0, 1.0, 2.0])
        y_points = torch.tensor([[0.0], [1.0], [2.0]])

        interp = LinearInterpolant(t_points, y_points)

        # Query at midpoints
        y_half = interp(0.5)
        assert torch.allclose(y_half, torch.tensor([0.5]), atol=1e-6)

        y_1_5 = interp(1.5)
        assert torch.allclose(y_1_5, torch.tensor([1.5]), atol=1e-6)

    def test_endpoints(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [1.0]])

        interp = LinearInterpolant(t_points, y_points)

        assert torch.allclose(interp(0.0), torch.tensor([0.0]), atol=1e-6)
        assert torch.allclose(interp(1.0), torch.tensor([1.0]), atol=1e-6)

    def test_multiple_queries(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [2.0]])

        interp = LinearInterpolant(t_points, y_points)

        t_query = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        y_query = interp(t_query)
        assert y_query.shape == (5, 1)
        expected = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]])
        assert torch.allclose(y_query, expected, atol=1e-6)

    def test_out_of_bounds_raises(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [1.0]])

        interp = LinearInterpolant(t_points, y_points)

        with pytest.raises(ValueError, match="outside"):
            interp(-0.1)

        with pytest.raises(ValueError, match="outside"):
            interp(1.1)

    def test_differentiable(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.tensor([[0.0], [1.0]], requires_grad=True)

        interp = LinearInterpolant(t_points, y_points)

        y_mid = interp(0.5)
        loss = y_mid.sum()
        loss.backward()

        assert y_points.grad is not None

    def test_batched_state(self):
        t_points = torch.tensor([0.0, 1.0])
        y_points = torch.randn(2, 3, 2)  # (T, B, D)

        interp = LinearInterpolant(t_points, y_points)

        # Query single time
        y_mid = interp(0.5)
        assert y_mid.shape == (3, 2)

        # Query multiple times
        t_query = torch.tensor([0.25, 0.5, 0.75])
        y_query = interp(t_query)
        assert y_query.shape == (3, 3, 2)  # (T_query, B, D)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__interpolant.py::TestLinearInterpolant -v`
Expected: FAIL with "ImportError"

**Step 3: Add LinearInterpolant implementation**

Add to `src/torchscience/integration/initial_value_problem/_interpolant.py`:

```python
class LinearInterpolant:
    """
    Linear interpolant for ODE dense output.

    Uses piecewise linear interpolation between grid points.
    The interpolant is differentiable and supports backpropagation.

    Parameters
    ----------
    t_points : Tensor
        Time points, shape (N,), must be monotonically increasing.
    y_points : Tensor
        State values at time points, shape (N, *state_shape).
    success : Tensor, optional
        Boolean mask indicating which batch elements succeeded.
        Shape (*batch_shape,). Only set when throw=False.
    """

    def __init__(
        self,
        t_points: torch.Tensor,
        y_points: torch.Tensor,
        success: Optional[torch.Tensor] = None,
    ):
        self.t_points = t_points
        self.y_points = y_points
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
        indices = torch.searchsorted(self.t_points, t.contiguous())
        indices = indices.clamp(1, len(self.t_points) - 1)

        # Get interval endpoints
        t0 = self.t_points[indices - 1]
        t1 = self.t_points[indices]
        y0 = self.y_points[indices - 1]
        y1 = self.y_points[indices]

        # Linear interpolation weight
        h = t1 - t0
        alpha = (t - t0) / h

        # Expand alpha for broadcasting with state dimensions
        state_dims = y0.dim() - 1
        for _ in range(state_dims):
            alpha = alpha.unsqueeze(-1)

        # Linear interpolation
        y = (1 - alpha) * y0 + alpha * y1

        if scalar_query:
            y = y.squeeze(0)

        return y
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__interpolant.py::TestLinearInterpolant -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/_interpolant.py tests/torchscience/integration/initial_value_problem/test__interpolant.py
git commit -m "feat(ivp): add LinearInterpolant for fixed-step solvers"
```

---

## Task 2: Euler Solver Implementation

**Files:**
- Create: `src/torchscience/integration/initial_value_problem/_euler.py`
- Create: `tests/torchscience/integration/initial_value_problem/test__euler.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/integration/initial_value_problem/test__euler.py
import pytest
import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem import euler


class TestEulerBasic:
    def test_exponential_decay(self):
        """Test against analytical solution: dy/dt = -y, y(0) = 1 => y(t) = exp(-t)"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = euler(decay, y0, t_span=(0.0, 1.0), dt=0.001)

        expected = torch.exp(torch.tensor([-1.0]))
        # Euler is 1st order, so we need small dt for good accuracy
        assert torch.allclose(y_final, expected, rtol=1e-2)

    def test_linear_ode(self):
        """dy/dt = 1 => y(t) = y0 + t (exact for Euler)"""

        def constant(t, y):
            return torch.ones_like(y)

        y0 = torch.tensor([0.0])
        y_final, interp = euler(constant, y0, t_span=(0.0, 1.0), dt=0.1)

        # Euler is exact for linear ODEs
        assert torch.allclose(y_final, torch.tensor([1.0]), atol=1e-6)

    def test_returns_interpolant(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_interpolant_endpoints(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert torch.allclose(interp(0.0), y0, atol=1e-6)
        assert torch.allclose(interp(1.0), y_final, atol=1e-6)


class TestEulerAutograd:
    def test_gradient_through_solver(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        y_final, _ = euler(f, y0, t_span=(0.0, 1.0), dt=0.01)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad).any()

    def test_gradient_through_interpolant(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        _, interp = euler(f, y0, t_span=(0.0, 1.0), dt=0.01)

        y_mid = interp(0.5)
        loss = y_mid.sum()
        loss.backward()

        assert theta.grad is not None


class TestEulerBatched:
    def test_batched_initial_conditions(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
        y_final, interp = euler(f, y0, t_span=(0.0, 1.0), dt=0.01)

        assert y_final.shape == (3, 1)

    def test_batched_interpolant(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0], [2.0]])  # (2, 1)
        _, interp = euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        t_query = torch.tensor([0.25, 0.5, 0.75])
        trajectory = interp(t_query)
        assert trajectory.shape == (3, 2, 1)  # (T, B, D)


class TestEulerTensorDict:
    def test_simple_tensordict(self):
        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict({"x": torch.tensor([1.0]), "v": torch.tensor([0.0])})
        state_final, interp = euler(f, state0, t_span=(0.0, 1.0), dt=0.01)

        assert isinstance(state_final, TensorDict)
        assert "x" in state_final.keys()
        assert "v" in state_final.keys()


class TestEulerComplex:
    def test_complex_exponential(self):
        """dy/dt = -i*y => y(t) = exp(-i*t)"""

        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = euler(f, y0, t_span=(0.0, 1.0), dt=0.001)

        expected = torch.exp(-1j * torch.tensor(1.0))
        assert torch.allclose(y_final.squeeze(), expected, atol=1e-2)


class TestEulerBackward:
    def test_backward_integration(self):
        """Integrate backwards"""

        def f(t, y):
            return -y

        y1 = torch.tensor([torch.exp(torch.tensor(-1.0))])
        y0_recovered, _ = euler(f, y1, t_span=(1.0, 0.0), dt=0.01)

        expected = torch.tensor([1.0])
        assert torch.allclose(y0_recovered, expected, rtol=0.1)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__euler.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/torchscience/integration/initial_value_problem/_euler.py
"""Forward Euler ODE solver."""

from typing import Callable, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem._interpolant import (
    LinearInterpolant,
)
from torchscience.integration.initial_value_problem._tensordict_utils import (
    flatten_state,
)


def euler(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    dt: float,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using forward Euler method.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
        Use closures or functools.partial to pass additional parameters.
    y0 : Tensor or TensorDict
        Initial state
    t_span : tuple[float, float]
        Integration interval (t0, t1). Supports backward integration if t1 < t0.
    dt : float
        Fixed step size (always positive; direction inferred from t_span)
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
    """
    t0, t1 = t_span
    direction = 1.0 if t1 >= t0 else -1.0
    h = direction * abs(dt)

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

    # Storage for interpolant
    t_points = [torch.tensor(t0, dtype=dtype, device=device)]
    y_points = [y_flat.clone()]

    t = t0
    y = y_flat.clone()

    while direction * (t1 - t) > 1e-10:
        # Clamp step to not overshoot
        h_step = h
        if direction * (t + h_step - t1) > 0:
            h_step = t1 - t

        # Euler step: y_{n+1} = y_n + h * f(t_n, y_n)
        dy = f_flat(t, y)
        y = y + h_step * dy
        t = t + h_step

        # Store for interpolant
        t_points.append(torch.tensor(t, dtype=dtype, device=device))
        y_points.append(y.clone())

    # Build interpolant
    t_tensor = torch.stack(t_points)
    y_tensor = torch.stack(y_points)

    success = None if throw else torch.ones(
        y_flat.shape[:-1] if y_flat.dim() > 1 else (),
        dtype=torch.bool,
        device=device,
    )

    interp = LinearInterpolant(t_tensor, y_tensor, success=success)

    # Wrap interpolant for TensorDict
    if is_tensordict:

        def interp_tensordict(t_query):
            y_flat_query = interp(t_query)
            if isinstance(t_query, (int, float)) or (
                isinstance(t_query, torch.Tensor) and t_query.dim() == 0
            ):
                return unflatten(y_flat_query)
            else:
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

**Step 4: Update module init**

Add to `src/torchscience/integration/initial_value_problem/__init__.py`:

```python
from torchscience.integration.initial_value_problem._euler import euler
```

And add `"euler"` to `__all__`.

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__euler.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/ tests/torchscience/integration/initial_value_problem/
git commit -m "feat(ivp): add euler fixed-step ODE solver"
```

---

## Task 3: Midpoint Solver Implementation

**Files:**
- Create: `src/torchscience/integration/initial_value_problem/_midpoint.py`
- Create: `tests/torchscience/integration/initial_value_problem/test__midpoint.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/integration/initial_value_problem/test__midpoint.py
import pytest
import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem import midpoint


class TestMidpointBasic:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 => y(t) = exp(-t)"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = midpoint(decay, y0, t_span=(0.0, 1.0), dt=0.01)

        expected = torch.exp(torch.tensor([-1.0]))
        # Midpoint is 2nd order, more accurate than Euler
        assert torch.allclose(y_final, expected, rtol=1e-3)

    def test_more_accurate_than_euler(self):
        """Midpoint should be more accurate than Euler for same step size"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        expected = torch.exp(torch.tensor([-1.0]))

        # Import euler for comparison
        from torchscience.integration.initial_value_problem import euler

        y_euler, _ = euler(decay, y0, t_span=(0.0, 1.0), dt=0.1)
        y_midpoint, _ = midpoint(decay, y0, t_span=(0.0, 1.0), dt=0.1)

        error_euler = (y_euler - expected).abs().item()
        error_midpoint = (y_midpoint - expected).abs().item()

        assert error_midpoint < error_euler

    def test_harmonic_oscillator(self):
        """Test 2D system: simple harmonic oscillator"""

        def oscillator(t, y):
            x, v = y[..., 0], y[..., 1]
            return torch.stack([v, -x], dim=-1)

        y0 = torch.tensor([1.0, 0.0])
        y_final, interp = midpoint(oscillator, y0, t_span=(0.0, 2 * torch.pi), dt=0.01)

        # After one period, should return near initial state
        assert torch.allclose(y_final, y0, atol=0.1)


class TestMidpointAutograd:
    def test_gradient_through_solver(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        y_final, _ = midpoint(f, y0, t_span=(0.0, 1.0), dt=0.01)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None


class TestMidpointTensorDict:
    def test_simple_tensordict(self):
        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict({"x": torch.tensor([1.0]), "v": torch.tensor([0.0])})
        state_final, interp = midpoint(f, state0, t_span=(0.0, 1.0), dt=0.01)

        assert isinstance(state_final, TensorDict)


class TestMidpointComplex:
    def test_complex_exponential(self):
        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = midpoint(f, y0, t_span=(0.0, 1.0), dt=0.01)

        expected = torch.exp(-1j * torch.tensor(1.0))
        assert torch.allclose(y_final.squeeze(), expected, atol=1e-3)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__midpoint.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/torchscience/integration/initial_value_problem/_midpoint.py
"""Midpoint (explicit) ODE solver."""

from typing import Callable, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem._interpolant import (
    LinearInterpolant,
)
from torchscience.integration.initial_value_problem._tensordict_utils import (
    flatten_state,
)


def midpoint(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    dt: float,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using explicit midpoint method (2nd order Runge-Kutta).

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
        Use closures or functools.partial to pass additional parameters.
    y0 : Tensor or TensorDict
        Initial state
    t_span : tuple[float, float]
        Integration interval (t0, t1). Supports backward integration if t1 < t0.
    dt : float
        Fixed step size (always positive; direction inferred from t_span)
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
    """
    t0, t1 = t_span
    direction = 1.0 if t1 >= t0 else -1.0
    h = direction * abs(dt)

    # Handle TensorDict
    is_tensordict = isinstance(y0, TensorDict)
    y_flat, unflatten = flatten_state(y0)

    if is_tensordict:

        def f_flat(t, y):
            y_struct = unflatten(y)
            dy_struct = f(t, y_struct)
            dy_flat, _ = flatten_state(dy_struct)
            return dy_flat

    else:
        f_flat = f

    dtype = y_flat.dtype
    device = y_flat.device

    t_points = [torch.tensor(t0, dtype=dtype, device=device)]
    y_points = [y_flat.clone()]

    t = t0
    y = y_flat.clone()

    while direction * (t1 - t) > 1e-10:
        h_step = h
        if direction * (t + h_step - t1) > 0:
            h_step = t1 - t

        # Midpoint method:
        # k1 = f(t_n, y_n)
        # k2 = f(t_n + h/2, y_n + h/2 * k1)
        # y_{n+1} = y_n + h * k2
        k1 = f_flat(t, y)
        k2 = f_flat(t + h_step / 2, y + h_step / 2 * k1)
        y = y + h_step * k2
        t = t + h_step

        t_points.append(torch.tensor(t, dtype=dtype, device=device))
        y_points.append(y.clone())

    t_tensor = torch.stack(t_points)
    y_tensor = torch.stack(y_points)

    success = None if throw else torch.ones(
        y_flat.shape[:-1] if y_flat.dim() > 1 else (),
        dtype=torch.bool,
        device=device,
    )

    interp = LinearInterpolant(t_tensor, y_tensor, success=success)

    if is_tensordict:

        def interp_tensordict(t_query):
            y_flat_query = interp(t_query)
            if isinstance(t_query, (int, float)) or (
                isinstance(t_query, torch.Tensor) and t_query.dim() == 0
            ):
                return unflatten(y_flat_query)
            else:
                return torch.stack(
                    [unflatten(y_flat_query[i]) for i in range(y_flat_query.shape[0])]
                )

        interp_tensordict.success = interp.success
        final_interp = interp_tensordict
    else:
        final_interp = interp

    if is_tensordict:
        y_final = unflatten(y)
    else:
        y_final = y

    return y_final, final_interp
```

**Step 4: Update module init**

Add to `src/torchscience/integration/initial_value_problem/__init__.py`:

```python
from torchscience.integration.initial_value_problem._midpoint import midpoint
```

And add `"midpoint"` to `__all__`.

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__midpoint.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/ tests/torchscience/integration/initial_value_problem/
git commit -m "feat(ivp): add midpoint 2nd-order fixed-step ODE solver"
```

---

## Task 4: Runge-Kutta 4 Solver Implementation

**Files:**
- Create: `src/torchscience/integration/initial_value_problem/_runge_kutta_4.py`
- Create: `tests/torchscience/integration/initial_value_problem/test__runge_kutta_4.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/integration/initial_value_problem/test__runge_kutta_4.py
import pytest
import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem import runge_kutta_4


class TestRungeKutta4Basic:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 => y(t) = exp(-t)"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = runge_kutta_4(decay, y0, t_span=(0.0, 1.0), dt=0.1)

        expected = torch.exp(torch.tensor([-1.0]))
        # RK4 is 4th order, very accurate
        assert torch.allclose(y_final, expected, rtol=1e-5)

    def test_more_accurate_than_midpoint(self):
        """RK4 should be more accurate than midpoint for same step size"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        expected = torch.exp(torch.tensor([-1.0]))

        from torchscience.integration.initial_value_problem import midpoint

        y_midpoint, _ = midpoint(decay, y0, t_span=(0.0, 1.0), dt=0.1)
        y_rk4, _ = runge_kutta_4(decay, y0, t_span=(0.0, 1.0), dt=0.1)

        error_midpoint = (y_midpoint - expected).abs().item()
        error_rk4 = (y_rk4 - expected).abs().item()

        assert error_rk4 < error_midpoint

    def test_harmonic_oscillator(self):
        """Test 2D system: simple harmonic oscillator"""

        def oscillator(t, y):
            x, v = y[..., 0], y[..., 1]
            return torch.stack([v, -x], dim=-1)

        y0 = torch.tensor([1.0, 0.0])
        y_final, interp = runge_kutta_4(
            oscillator, y0, t_span=(0.0, 2 * torch.pi), dt=0.1
        )

        # After one period, should return close to initial state
        assert torch.allclose(y_final, y0, atol=1e-3)

    def test_interpolant_trajectory(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        _, interp = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        t_query = torch.linspace(0, 1, 20)
        trajectory = interp(t_query)
        assert trajectory.shape == (20, 1)

        # Should be monotonically decreasing
        for i in range(19):
            assert trajectory[i, 0] > trajectory[i + 1, 0]


class TestRungeKutta4Autograd:
    def test_gradient_through_solver(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        y_final, _ = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None

    def test_gradcheck(self):
        """Verify gradients are correct"""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def solve(y0):
            y_final, _ = runge_kutta_4(f, y0, t_span=(0.0, 0.5), dt=0.1)
            return y_final

        assert torch.autograd.gradcheck(solve, (y0,), raise_exception=True)


class TestRungeKutta4TensorDict:
    def test_simple_tensordict(self):
        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict({"x": torch.tensor([1.0]), "v": torch.tensor([0.0])})
        state_final, _ = runge_kutta_4(f, state0, t_span=(0.0, 1.0), dt=0.1)

        assert isinstance(state_final, TensorDict)


class TestRungeKutta4Complex:
    def test_complex_exponential(self):
        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        expected = torch.exp(-1j * torch.tensor(1.0))
        assert torch.allclose(y_final.squeeze(), expected, atol=1e-5)


class TestRungeKutta4SciPy:
    def test_matches_scipy(self):
        scipy = pytest.importorskip("scipy")
        from scipy.integrate import solve_ivp

        def f_torch(t, y):
            return -y

        def f_scipy(t, y):
            return -y

        y0_val = 1.0
        t_span = (0.0, 2.0)

        y0_torch = torch.tensor([y0_val], dtype=torch.float64)
        y_torch, _ = runge_kutta_4(f_torch, y0_torch, t_span, dt=0.01)

        sol_scipy = solve_ivp(f_scipy, t_span, [y0_val], method="RK45", max_step=0.01)

        assert torch.allclose(
            y_torch, torch.tensor(sol_scipy.y[:, -1], dtype=torch.float64), rtol=1e-3
        )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__runge_kutta_4.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/torchscience/integration/initial_value_problem/_runge_kutta_4.py
"""Classic 4th-order Runge-Kutta ODE solver."""

from typing import Callable, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem._interpolant import (
    LinearInterpolant,
)
from torchscience.integration.initial_value_problem._tensordict_utils import (
    flatten_state,
)


def runge_kutta_4(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    dt: float,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using classic 4th-order Runge-Kutta method (RK4).

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
        Use closures or functools.partial to pass additional parameters.
    y0 : Tensor or TensorDict
        Initial state
    t_span : tuple[float, float]
        Integration interval (t0, t1). Supports backward integration if t1 < t0.
    dt : float
        Fixed step size (always positive; direction inferred from t_span)
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
    """
    t0, t1 = t_span
    direction = 1.0 if t1 >= t0 else -1.0
    h = direction * abs(dt)

    # Handle TensorDict
    is_tensordict = isinstance(y0, TensorDict)
    y_flat, unflatten = flatten_state(y0)

    if is_tensordict:

        def f_flat(t, y):
            y_struct = unflatten(y)
            dy_struct = f(t, y_struct)
            dy_flat, _ = flatten_state(dy_struct)
            return dy_flat

    else:
        f_flat = f

    dtype = y_flat.dtype
    device = y_flat.device

    t_points = [torch.tensor(t0, dtype=dtype, device=device)]
    y_points = [y_flat.clone()]

    t = t0
    y = y_flat.clone()

    while direction * (t1 - t) > 1e-10:
        h_step = h
        if direction * (t + h_step - t1) > 0:
            h_step = t1 - t

        # Classic RK4:
        # k1 = f(t_n, y_n)
        # k2 = f(t_n + h/2, y_n + h/2 * k1)
        # k3 = f(t_n + h/2, y_n + h/2 * k2)
        # k4 = f(t_n + h, y_n + h * k3)
        # y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        k1 = f_flat(t, y)
        k2 = f_flat(t + h_step / 2, y + h_step / 2 * k1)
        k3 = f_flat(t + h_step / 2, y + h_step / 2 * k2)
        k4 = f_flat(t + h_step, y + h_step * k3)

        y = y + h_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h_step

        t_points.append(torch.tensor(t, dtype=dtype, device=device))
        y_points.append(y.clone())

    t_tensor = torch.stack(t_points)
    y_tensor = torch.stack(y_points)

    success = None if throw else torch.ones(
        y_flat.shape[:-1] if y_flat.dim() > 1 else (),
        dtype=torch.bool,
        device=device,
    )

    interp = LinearInterpolant(t_tensor, y_tensor, success=success)

    if is_tensordict:

        def interp_tensordict(t_query):
            y_flat_query = interp(t_query)
            if isinstance(t_query, (int, float)) or (
                isinstance(t_query, torch.Tensor) and t_query.dim() == 0
            ):
                return unflatten(y_flat_query)
            else:
                return torch.stack(
                    [unflatten(y_flat_query[i]) for i in range(y_flat_query.shape[0])]
                )

        interp_tensordict.success = interp.success
        final_interp = interp_tensordict
    else:
        final_interp = interp

    if is_tensordict:
        y_final = unflatten(y)
    else:
        y_final = y

    return y_final, final_interp
```

**Step 4: Update module init**

Add to `src/torchscience/integration/initial_value_problem/__init__.py`:

```python
from torchscience.integration.initial_value_problem._runge_kutta_4 import runge_kutta_4
```

And add `"runge_kutta_4"` to `__all__`.

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__runge_kutta_4.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/ tests/torchscience/integration/initial_value_problem/
git commit -m "feat(ivp): add runge_kutta_4 4th-order fixed-step ODE solver"
```

---

## Task 5: CPU C++ Kernels for Fixed-Step Solvers

**Files:**
- Create: `src/torchscience/csrc/cpu/integration/initial_value_problem/euler.h`
- Create: `src/torchscience/csrc/cpu/integration/initial_value_problem/midpoint.h`
- Create: `src/torchscience/csrc/cpu/integration/initial_value_problem/runge_kutta_4.h`

**Step 1: Write Euler kernel header**

```cpp
// src/torchscience/csrc/cpu/integration/initial_value_problem/euler.h
#pragma once

#include <ATen/ATen.h>
#include <functional>

namespace torchscience {
namespace cpu {
namespace integration {
namespace initial_value_problem {

template <typename scalar_t>
struct EulerStep {
  // Single step of forward Euler method
  static at::Tensor step(
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& f,
      scalar_t t,
      const at::Tensor& y,
      scalar_t h
  ) {
    at::Tensor k = f(t, y);
    return y + h * k;
  }
};

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cpu
}  // namespace torchscience
```

**Step 2: Write Midpoint kernel header**

```cpp
// src/torchscience/csrc/cpu/integration/initial_value_problem/midpoint.h
#pragma once

#include <ATen/ATen.h>
#include <functional>

namespace torchscience {
namespace cpu {
namespace integration {
namespace initial_value_problem {

template <typename scalar_t>
struct MidpointStep {
  // Single step of explicit midpoint method (RK2)
  static at::Tensor step(
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& f,
      scalar_t t,
      const at::Tensor& y,
      scalar_t h
  ) {
    at::Tensor k1 = f(t, y);
    at::Tensor k2 = f(t + h / 2, y + h / 2 * k1);
    return y + h * k2;
  }
};

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cpu
}  // namespace torchscience
```

**Step 3: Write RK4 kernel header**

```cpp
// src/torchscience/csrc/cpu/integration/initial_value_problem/runge_kutta_4.h
#pragma once

#include <ATen/ATen.h>
#include <functional>

namespace torchscience {
namespace cpu {
namespace integration {
namespace initial_value_problem {

template <typename scalar_t>
struct RungeKutta4Step {
  // Single step of classic 4th-order Runge-Kutta
  static at::Tensor step(
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& f,
      scalar_t t,
      const at::Tensor& y,
      scalar_t h
  ) {
    at::Tensor k1 = f(t, y);
    at::Tensor k2 = f(t + h / 2, y + h / 2 * k1);
    at::Tensor k3 = f(t + h / 2, y + h / 2 * k2);
    at::Tensor k4 = f(t + h, y + h * k3);
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
  }
};

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cpu
}  // namespace torchscience
```

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cpu/integration/initial_value_problem/
git commit -m "feat(ivp): add CPU kernel headers for fixed-step solvers"
```

---

## Task 6: CUDA Kernel Placeholders for Fixed-Step Solvers

**Files:**
- Create: `src/torchscience/csrc/cuda/integration/initial_value_problem/euler.cu`
- Create: `src/torchscience/csrc/cuda/integration/initial_value_problem/midpoint.cu`
- Create: `src/torchscience/csrc/cuda/integration/initial_value_problem/runge_kutta_4.cu`

**Step 1: Write Euler CUDA placeholder**

```cpp
// src/torchscience/csrc/cuda/integration/initial_value_problem/euler.cu
#include <ATen/ATen.h>

namespace torchscience {
namespace cuda {
namespace integration {
namespace initial_value_problem {

// TODO: Implement CUDA kernel for Euler method
// For now, the Python implementation handles CUDA tensors.

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cuda
}  // namespace torchscience
```

**Step 2: Write similar placeholders for midpoint and runge_kutta_4**

(Similar structure)

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cuda/integration/initial_value_problem/
git commit -m "feat(ivp): add CUDA kernel placeholders for fixed-step solvers"
```

---

## Task 7: Comparative Tests Across Solvers

**Files:**
- Create: `tests/torchscience/integration/initial_value_problem/test__solvers_comparison.py`

**Step 1: Write comparative tests**

```python
# tests/torchscience/integration/initial_value_problem/test__solvers_comparison.py
import pytest
import torch

from torchscience.integration.initial_value_problem import (
    dormand_prince_5,
    euler,
    midpoint,
    runge_kutta_4,
)


class TestSolverAccuracyOrdering:
    """Verify that higher-order methods are more accurate for same step size."""

    def test_accuracy_ordering_exponential_decay(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        dt = 0.1

        y_euler, _ = euler(f, y0, t_span=(0.0, 1.0), dt=dt)
        y_midpoint, _ = midpoint(f, y0, t_span=(0.0, 1.0), dt=dt)
        y_rk4, _ = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=dt)
        y_dp5, _ = dormand_prince_5(f, y0, t_span=(0.0, 1.0), rtol=1e-8, atol=1e-10)

        error_euler = (y_euler - expected).abs().item()
        error_midpoint = (y_midpoint - expected).abs().item()
        error_rk4 = (y_rk4 - expected).abs().item()
        error_dp5 = (y_dp5 - expected).abs().item()

        # Higher order => smaller error
        assert error_midpoint < error_euler, "Midpoint should beat Euler"
        assert error_rk4 < error_midpoint, "RK4 should beat Midpoint"
        assert error_dp5 < error_rk4 * 10, "DP5 should be very accurate"


class TestSolverConsistency:
    """Verify all solvers produce consistent results for simple problems."""

    def test_all_solvers_converge_to_same_solution(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        # Use small enough dt that all converge well
        y_euler, _ = euler(f, y0, t_span=(0.0, 1.0), dt=0.001)
        y_midpoint, _ = midpoint(f, y0, t_span=(0.0, 1.0), dt=0.001)
        y_rk4, _ = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=0.001)
        y_dp5, _ = dormand_prince_5(f, y0, t_span=(0.0, 1.0), rtol=1e-10, atol=1e-12)

        # All should be close to expected
        assert torch.allclose(y_euler, expected, rtol=1e-2)
        assert torch.allclose(y_midpoint, expected, rtol=1e-4)
        assert torch.allclose(y_rk4, expected, rtol=1e-6)
        assert torch.allclose(y_dp5, expected, rtol=1e-8)


class TestSolverInterpolantConsistency:
    """Verify interpolants work consistently across solvers."""

    def test_interpolant_endpoints_all_solvers(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])

        solvers = [
            ("euler", lambda: euler(f, y0, (0.0, 1.0), dt=0.1)),
            ("midpoint", lambda: midpoint(f, y0, (0.0, 1.0), dt=0.1)),
            ("runge_kutta_4", lambda: runge_kutta_4(f, y0, (0.0, 1.0), dt=0.1)),
            ("dormand_prince_5", lambda: dormand_prince_5(f, y0, (0.0, 1.0))),
        ]

        for name, solve in solvers:
            y_final, interp = solve()

            # Start point should match y0
            assert torch.allclose(
                interp(0.0), y0, atol=1e-5
            ), f"{name}: start mismatch"

            # End point should match y_final
            assert torch.allclose(
                interp(1.0), y_final, atol=1e-5
            ), f"{name}: end mismatch"
```

**Step 2: Run tests**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__solvers_comparison.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/torchscience/integration/initial_value_problem/test__solvers_comparison.py
git commit -m "test(ivp): add comparative tests across all explicit solvers"
```

---

## Task 8: Final Module Exports

**Files:**
- Modify: `src/torchscience/integration/initial_value_problem/__init__.py`

**Step 1: Update with complete exports and docstring**

```python
# src/torchscience/integration/initial_value_problem/__init__.py
"""
Initial value problem solvers for ordinary differential equations.

This module provides differentiable ODE solvers for PyTorch tensors and TensorDict.

Available Solvers
-----------------
euler
    Forward Euler method (1st order, fixed step).
    Simplest method, educational baseline.

midpoint
    Explicit midpoint method (2nd order, fixed step).
    Good accuracy/cost tradeoff for smooth problems.

runge_kutta_4
    Classic 4th-order Runge-Kutta (fixed step).
    Widely used workhorse, excellent for non-stiff problems.

dormand_prince_5
    Dormand-Prince 5(4) adaptive method.
    Production-quality solver with error control.

Examples
--------
>>> import torch
>>> from torchscience.integration.initial_value_problem import runge_kutta_4
>>>
>>> def decay(t, y):
...     return -y
>>>
>>> y0 = torch.tensor([1.0])
>>> y_final, interp = runge_kutta_4(decay, y0, t_span=(0.0, 5.0), dt=0.01)
>>> trajectory = interp(torch.linspace(0, 5, 100))
"""

from torchscience.integration.initial_value_problem._dormand_prince_5 import (
    dormand_prince_5,
)
from torchscience.integration.initial_value_problem._euler import euler
from torchscience.integration.initial_value_problem._exceptions import (
    ConvergenceError,
    MaxStepsExceeded,
    ODESolverError,
    StepSizeError,
)
from torchscience.integration.initial_value_problem._midpoint import midpoint
from torchscience.integration.initial_value_problem._runge_kutta_4 import runge_kutta_4

__all__ = [
    # Exceptions
    "ConvergenceError",
    "MaxStepsExceeded",
    "ODESolverError",
    "StepSizeError",
    # Solvers (ordered by complexity)
    "euler",
    "midpoint",
    "runge_kutta_4",
    "dormand_prince_5",
]
```

**Step 2: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/__init__.py
git commit -m "docs(ivp): complete Phase 2 with all fixed-step explicit solvers"
```

---

## Summary

Phase 2 implements three fixed-step explicit ODE solvers:

1. **LinearInterpolant** - Piecewise linear interpolation for fixed-step solvers
2. **euler** - Forward Euler (1st order)
3. **midpoint** - Explicit midpoint (2nd order Runge-Kutta)
4. **runge_kutta_4** - Classic RK4 (4th order)
5. **CPU C++ kernel headers** - Template implementations
6. **CUDA kernel placeholders** - For future GPU acceleration
7. **Comparative tests** - Verify accuracy ordering across methods

All solvers share the same API and features:
- TensorDict support
- Complex number support
- Autograd support
- Backward integration
- Dense output via interpolant

Phase 3 (backward_euler) builds on this foundation to add implicit solvers.
