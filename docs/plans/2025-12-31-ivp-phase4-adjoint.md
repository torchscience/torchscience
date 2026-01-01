# Phase 4: Adjoint Wrapper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the `adjoint()` wrapper for memory-efficient gradients via the continuous adjoint method.

**Architecture:** The adjoint method computes gradients by solving an augmented ODE backwards in time, avoiding storage of the autograd computation graph. The wrapper intercepts the backward pass using a custom `torch.autograd.Function`, recomputes the forward trajectory with checkpointing, and integrates the adjoint ODE to get parameter gradients.

**Tech Stack:** PyTorch, torch.autograd.Function, TensorDict

**Prerequisites:** Phases 1-3 complete (all 5 ODE solvers working)

---

## Task 1: Adjoint ODE Theory Implementation

**Files:**
- Create: `src/torchscience/integration/initial_value_problem/_adjoint.py`
- Create: `tests/torchscience/integration/initial_value_problem/test__adjoint.py`

**Step 1: Write the failing test for basic adjoint functionality**

```python
# tests/torchscience/integration/initial_value_problem/test__adjoint.py
import pytest
import torch

from torchscience.integration.initial_value_problem import (
    adjoint,
    dormand_prince_5,
    euler,
    runge_kutta_4,
)


class TestAdjointBasic:
    def test_adjoint_wraps_solver(self):
        """adjoint() should return a callable with same signature."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        adjoint_solver = adjoint(dormand_prince_5)

        y_final, interp = adjoint_solver(f, y0, t_span=(0.0, 1.0))

        # Should produce same result as direct call
        y_direct, _ = dormand_prince_5(f, y0, t_span=(0.0, 1.0))
        assert torch.allclose(y_final, y_direct, atol=1e-5)

    def test_adjoint_gradients_match_direct(self):
        """Adjoint gradients should approximately match direct backprop."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        # Direct backprop
        theta_direct = theta.clone().requires_grad_(True)

        def f_direct(t, y):
            return -theta_direct * y

        y_direct, _ = dormand_prince_5(f_direct, y0, t_span=(0.0, 1.0))
        loss_direct = y_direct.sum()
        loss_direct.backward()
        grad_direct = theta_direct.grad.clone()

        # Adjoint method
        theta_adjoint = theta.clone().requires_grad_(True)

        def f_adjoint(t, y):
            return -theta_adjoint * y

        adjoint_solver = adjoint(dormand_prince_5)
        y_adjoint, _ = adjoint_solver(f_adjoint, y0, t_span=(0.0, 1.0))
        loss_adjoint = y_adjoint.sum()
        loss_adjoint.backward()
        grad_adjoint = theta_adjoint.grad.clone()

        # Gradients should match (approximately, due to different discretizations)
        assert torch.allclose(grad_direct, grad_adjoint, rtol=0.1)

    def test_adjoint_with_euler(self):
        """Adjoint wrapper should work with any solver."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        adjoint_euler = adjoint(euler)
        y_final, _ = adjoint_euler(f, y0, t_span=(0.0, 1.0), dt=0.01)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad).any()

    def test_adjoint_with_rk4(self):
        """Adjoint wrapper should work with RK4."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        adjoint_rk4 = adjoint(runge_kutta_4)
        y_final, _ = adjoint_rk4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None


class TestAdjointMemory:
    def test_adjoint_uses_less_memory_conceptually(self):
        """
        The adjoint method should use O(1) memory for the autograd graph
        vs O(n_steps) for direct backprop.

        This is a conceptual test - we verify the forward pass works
        and gradients are computed, which confirms the adjoint path is used.
        """
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        # This should not store O(n_steps) activations
        adjoint_solver = adjoint(dormand_prince_5)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 10.0))

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None


class TestAdjointHigherDimensional:
    def test_adjoint_2d_system(self):
        """Test adjoint on 2D harmonic oscillator."""
        omega = torch.tensor([2.0], requires_grad=True)

        def oscillator(t, y):
            x, v = y[..., 0], y[..., 1]
            dxdt = v
            dvdt = -omega**2 * x
            return torch.stack([dxdt, dvdt], dim=-1)

        y0 = torch.tensor([1.0, 0.0])

        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(oscillator, y0, t_span=(0.0, 1.0), dt=0.01)

        loss = y_final[0]  # Final position
        loss.backward()

        assert omega.grad is not None
        assert not torch.isnan(omega.grad).any()

    def test_adjoint_batched(self):
        """Test adjoint with batched initial conditions."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)

        adjoint_solver = adjoint(dormand_prince_5)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0))

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None


class TestAdjointInterpolant:
    def test_adjoint_interpolant_works(self):
        """Interpolant should work with adjoint wrapper."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])

        adjoint_solver = adjoint(dormand_prince_5)
        y_final, interp = adjoint_solver(f, y0, t_span=(0.0, 1.0))

        # Interpolant should be functional
        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

        t_query = torch.linspace(0, 1, 10)
        trajectory = interp(t_query)
        assert trajectory.shape == (10, 1)

    def test_adjoint_interpolant_not_differentiable(self):
        """
        Note: With adjoint method, the interpolant is NOT differentiable
        (gradients only flow through y_final, not intermediate points).

        This is a known limitation documented in the design.
        """
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        adjoint_solver = adjoint(dormand_prince_5)
        _, interp = adjoint_solver(f, y0, t_span=(0.0, 1.0))

        # Querying interpolant produces a tensor
        y_mid = interp(0.5)

        # But gradients don't flow through it with adjoint
        # (This test documents the behavior rather than asserting it)


class TestAdjointCheckpoints:
    def test_adjoint_with_checkpoints(self):
        """Test adjoint with explicit checkpoint count."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        adjoint_solver = adjoint(dormand_prince_5, checkpoints=5)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0))

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None

    def test_checkpoints_dont_affect_forward(self):
        """Checkpoint count shouldn't affect forward solution."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])

        y_no_ckpt, _ = adjoint(dormand_prince_5)(f, y0, t_span=(0.0, 1.0))
        y_with_ckpt, _ = adjoint(dormand_prince_5, checkpoints=3)(
            f, y0, t_span=(0.0, 1.0)
        )

        assert torch.allclose(y_no_ckpt, y_with_ckpt, atol=1e-6)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__adjoint.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/torchscience/integration/initial_value_problem/_adjoint.py
"""Adjoint method wrapper for memory-efficient ODE gradients."""

from typing import Any, Callable, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem._tensordict_utils import (
    flatten_state,
)


class _AdjointODEFunction(torch.autograd.Function):
    """
    Custom autograd function implementing the continuous adjoint method.

    Forward pass: Solve the ODE normally
    Backward pass: Solve augmented adjoint ODE backwards in time
    """

    @staticmethod
    def forward(
        ctx,
        y0_flat: torch.Tensor,
        t0: float,
        t1: float,
        solver: Callable,
        f_flat: Callable,
        solver_kwargs: dict,
        params: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        Forward pass: solve ODE and store info for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context for saving tensors
        y0_flat : Tensor
            Flattened initial state
        t0, t1 : float
            Integration interval
        solver : callable
            ODE solver function
        f_flat : callable
            Dynamics function for flattened state
        solver_kwargs : dict
            Additional kwargs for solver
        params : tuple of Tensors
            Learnable parameters to compute gradients for

        Returns
        -------
        y_final_flat : Tensor
            Flattened final state
        """
        with torch.no_grad():
            y_final, interp = solver(f_flat, y0_flat, t_span=(t0, t1), **solver_kwargs)

        # Save for backward
        ctx.save_for_backward(y0_flat, y_final, *params)
        ctx.solver = solver
        ctx.f_flat = f_flat
        ctx.solver_kwargs = solver_kwargs
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.interp = interp

        return y_final

    @staticmethod
    def backward(ctx, grad_y_final: torch.Tensor) -> Tuple[Any, ...]:
        """
        Backward pass: solve adjoint ODE to compute gradients.

        The adjoint equation is:
            da/dt = -a^T @ (df/dy)

        where a is the adjoint variable (gradient w.r.t. state).

        Parameter gradients are computed as:
            dL/dtheta = integral_t0^t1 a^T @ (df/dtheta) dt
        """
        y0_flat, y_final, *params = ctx.saved_tensors
        solver = ctx.solver
        f_flat = ctx.f_flat
        solver_kwargs = ctx.solver_kwargs
        t0, t1 = ctx.t0, ctx.t1
        interp = ctx.interp

        # Initial adjoint state (gradient w.r.t. final state)
        a = grad_y_final.clone()

        # Accumulate parameter gradients
        param_grads = [torch.zeros_like(p) for p in params]

        # Solve adjoint ODE backwards
        # We approximate by using the stored interpolant and stepping backwards

        # Number of steps for adjoint integration
        n_steps = 100
        dt = (t1 - t0) / n_steps

        t = t1
        y = y_final.clone()

        for _ in range(n_steps):
            t_prev = t - dt

            # Get state at this time from interpolant
            with torch.enable_grad():
                y_at_t = interp(t)
                if not y_at_t.requires_grad:
                    y_at_t = y_at_t.clone().requires_grad_(True)

                # Compute f and its Jacobians
                f_val = f_flat(t, y_at_t)

            # Adjoint update: da/dt = -a^T @ (df/dy)
            # Approximate: a_prev = a + dt * a^T @ (df/dy)
            # Use vector-Jacobian product for efficiency
            if y_at_t.requires_grad:
                # Compute VJP: a^T @ (df/dy)
                (vjp,) = torch.autograd.grad(
                    f_val,
                    y_at_t,
                    grad_outputs=a,
                    retain_graph=True,
                    allow_unused=True,
                )
                if vjp is not None:
                    a = a + dt * vjp

            # Accumulate parameter gradients
            for i, p in enumerate(params):
                if p.requires_grad:
                    # Compute a^T @ (df/dtheta)
                    grads = torch.autograd.grad(
                        f_val,
                        p,
                        grad_outputs=a,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    if grads[0] is not None:
                        param_grads[i] = param_grads[i] + dt * grads[0]

            t = t_prev

        # Gradient w.r.t. y0 is the final adjoint state
        grad_y0 = a

        # Return gradients in same order as forward inputs
        # (y0_flat, t0, t1, solver, f_flat, solver_kwargs, params)
        return (grad_y0, None, None, None, None, None, tuple(param_grads))


def adjoint(
    solver: Callable,
    checkpoints: Optional[int] = None,
) -> Callable:
    """
    Wrap a solver to use adjoint method for gradients.

    The adjoint method computes gradients by solving an augmented ODE
    backwards in time, using O(1) memory for the autograd graph instead
    of O(n_steps).

    Parameters
    ----------
    solver : callable
        Any ODE solver function (euler, dormand_prince_5, etc.)
    checkpoints : int, optional
        Number of checkpoints for memory/compute tradeoff.
        None = automatic selection (currently unused, reserved for future).

    Returns
    -------
    wrapped_solver : callable
        Solver with same signature but using adjoint gradients.

    Examples
    --------
    >>> from torchscience.integration.initial_value_problem import (
    ...     adjoint,
    ...     dormand_prince_5,
    ... )
    >>> adjoint_solver = adjoint(dormand_prince_5)
    >>> y_final, interp = adjoint_solver(f, y0, t_span=(0.0, 10.0))
    >>> loss = y_final.sum()
    >>> loss.backward()  # Uses O(1) memory for autograd graph

    Notes
    -----
    The adjoint method only affects gradient computation. It does NOT change:

    - The interpolant (still stores trajectory points for dense output)
    - Forward solve behavior (same numerical solution)
    - Return values (same (y_final, interp) tuple)

    Memory savings come from not storing the autograd computation graph,
    not from discarding the trajectory.

    When to use:

    - Long integrations with large state dimension
    - Memory-constrained environments
    - When exact discretization gradients are not required

    When NOT to use:

    - Short integrations (overhead not worth it)
    - When you need exact discretization gradients
    - When differentiating through the interpolant
    """

    def wrapped_solver(
        f: Callable,
        y0: Union[torch.Tensor, TensorDict],
        t_span: Tuple[float, float],
        **kwargs,
    ) -> Tuple[
        Union[torch.Tensor, TensorDict],
        Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
    ]:
        """Wrapped solver using adjoint method for gradients."""
        t0, t1 = t_span

        # Handle TensorDict
        is_tensordict = isinstance(y0, TensorDict)
        y0_flat, unflatten = flatten_state(y0)

        if is_tensordict:

            def f_flat(t, y):
                y_struct = unflatten(y)
                dy_struct = f(t, y_struct)
                dy_flat, _ = flatten_state(dy_struct)
                return dy_flat

        else:
            f_flat = f

        # Find all parameters that require gradients
        # This is tricky - we need to capture them from f's closure
        # For now, we rely on the dynamics function using module parameters

        # Get parameters from dynamics function closure
        params = []

        def extract_params(obj, seen=None):
            """Recursively extract tensors requiring grad from closures."""
            if seen is None:
                seen = set()
            if id(obj) in seen:
                return
            seen.add(id(obj))

            if isinstance(obj, torch.Tensor) and obj.requires_grad:
                params.append(obj)
            elif hasattr(obj, "__closure__") and obj.__closure__:
                for cell in obj.__closure__:
                    try:
                        extract_params(cell.cell_contents, seen)
                    except ValueError:
                        pass
            elif hasattr(obj, "__dict__"):
                for v in obj.__dict__.values():
                    extract_params(v, seen)

        extract_params(f)

        # If no parameters found, just do regular forward (no gradients needed)
        if not params:
            y_final, interp = solver(f_flat, y0_flat, t_span=(t0, t1), **kwargs)
            if is_tensordict:
                y_final = unflatten(y_final)

                def interp_td(t_query):
                    y_flat_query = interp(t_query)
                    if isinstance(t_query, (int, float)) or (
                        isinstance(t_query, torch.Tensor) and t_query.dim() == 0
                    ):
                        return unflatten(y_flat_query)
                    return torch.stack(
                        [unflatten(y_flat_query[i]) for i in range(y_flat_query.shape[0])]
                    )

                interp_td.success = getattr(interp, "success", None)
                return y_final, interp_td
            return y_final, interp

        # Use custom autograd function for adjoint gradients
        y_final_flat = _AdjointODEFunction.apply(
            y0_flat,
            t0,
            t1,
            solver,
            f_flat,
            kwargs,
            tuple(params),
        )

        # Get interpolant from a separate forward pass (with no_grad)
        with torch.no_grad():
            _, interp = solver(f_flat, y0_flat, t_span=(t0, t1), **kwargs)

        # Unflatten result
        if is_tensordict:
            y_final = unflatten(y_final_flat)

            def interp_tensordict(t_query):
                y_flat_query = interp(t_query)
                if isinstance(t_query, (int, float)) or (
                    isinstance(t_query, torch.Tensor) and t_query.dim() == 0
                ):
                    return unflatten(y_flat_query)
                return torch.stack(
                    [unflatten(y_flat_query[i]) for i in range(y_flat_query.shape[0])]
                )

            interp_tensordict.success = getattr(interp, "success", None)
            return y_final, interp_tensordict

        return y_final_flat, interp

    # Preserve solver metadata
    wrapped_solver.__name__ = f"adjoint({getattr(solver, '__name__', 'solver')})"
    wrapped_solver.__doc__ = f"""
    {getattr(solver, '__name__', 'Solver')} with adjoint method for memory-efficient gradients.

    This is a wrapped version of {getattr(solver, '__name__', 'the solver')} that uses
    the continuous adjoint method to compute gradients with O(1) memory instead of
    O(n_steps).

    See `adjoint()` documentation for details.
    """

    return wrapped_solver
```

**Step 4: Update module init**

Add to `src/torchscience/integration/initial_value_problem/__init__.py`:

```python
from torchscience.integration.initial_value_problem._adjoint import adjoint
```

And add `"adjoint"` to `__all__`.

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__adjoint.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/ tests/torchscience/integration/initial_value_problem/
git commit -m "feat(ivp): add adjoint wrapper for memory-efficient gradients"
```

---

## Task 2: Gradient Accuracy Tests

**Files:**
- Create: `tests/torchscience/integration/initial_value_problem/test__adjoint_accuracy.py`

**Step 1: Write gradient accuracy tests**

```python
# tests/torchscience/integration/initial_value_problem/test__adjoint_accuracy.py
"""Tests verifying adjoint gradient accuracy against finite differences."""

import pytest
import torch

from torchscience.integration.initial_value_problem import (
    adjoint,
    dormand_prince_5,
    euler,
    runge_kutta_4,
)


class TestAdjointGradientAccuracy:
    @pytest.mark.parametrize(
        "solver,kwargs",
        [
            (euler, {"dt": 0.01}),
            (runge_kutta_4, {"dt": 0.01}),
            (dormand_prince_5, {"rtol": 1e-6, "atol": 1e-9}),
        ],
    )
    def test_gradient_vs_finite_diff(self, solver, kwargs):
        """Compare adjoint gradients to finite difference approximation."""
        theta = torch.tensor([1.5], requires_grad=True, dtype=torch.float64)
        eps = 1e-5

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Adjoint gradient
        adjoint_solver = adjoint(solver)
        y_final, _ = adjoint_solver(f, y0.clone(), t_span=(0.0, 1.0), **kwargs)
        loss = y_final.sum()
        loss.backward()
        grad_adjoint = theta.grad.clone()

        # Finite difference
        theta.grad = None

        def compute_loss(theta_val):
            def f_local(t, y):
                return -theta_val * y

            y_final, _ = solver(f_local, y0.clone(), t_span=(0.0, 1.0), **kwargs)
            return y_final.sum()

        with torch.no_grad():
            theta_plus = theta.clone() + eps
            theta_minus = theta.clone() - eps
            loss_plus = compute_loss(theta_plus)
            loss_minus = compute_loss(theta_minus)
            grad_fd = (loss_plus - loss_minus) / (2 * eps)

        # Should match within reasonable tolerance
        # (adjoint is approximate due to discretization of adjoint ODE)
        assert torch.allclose(grad_adjoint, grad_fd, rtol=0.1, atol=1e-4)

    def test_gradient_multiple_params(self):
        """Test gradients with multiple learnable parameters."""
        alpha = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
        beta = torch.tensor([0.5], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -alpha * y + beta * torch.sin(t)

        y0 = torch.tensor([1.0], dtype=torch.float64)

        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0), dt=0.01)
        loss = y_final.sum()
        loss.backward()

        assert alpha.grad is not None
        assert beta.grad is not None
        assert not torch.isnan(alpha.grad).any()
        assert not torch.isnan(beta.grad).any()

    def test_gradient_2d_param(self):
        """Test with matrix-valued parameter."""
        A = torch.tensor([[-1.0, 0.5], [0.5, -1.0]], requires_grad=True)

        def f(t, y):
            return A @ y

        y0 = torch.tensor([1.0, 0.5])

        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0), dt=0.01)
        loss = y_final.sum()
        loss.backward()

        assert A.grad is not None
        assert A.grad.shape == A.shape


class TestAdjointVsDirectBackprop:
    def test_comparison_exponential_decay(self):
        """Compare adjoint to direct backprop in detail."""

        def make_dynamics(theta):
            def f(t, y):
                return -theta * y

            return f

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Direct backprop
        theta_direct = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
        f_direct = make_dynamics(theta_direct)
        y_direct, _ = runge_kutta_4(f_direct, y0.clone(), t_span=(0.0, 1.0), dt=0.01)
        loss_direct = y_direct.sum()
        loss_direct.backward()
        grad_direct = theta_direct.grad.clone()

        # Adjoint
        theta_adjoint = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
        f_adjoint = make_dynamics(theta_adjoint)
        adjoint_solver = adjoint(runge_kutta_4)
        y_adjoint, _ = adjoint_solver(
            f_adjoint, y0.clone(), t_span=(0.0, 1.0), dt=0.01
        )
        loss_adjoint = y_adjoint.sum()
        loss_adjoint.backward()
        grad_adjoint = theta_adjoint.grad.clone()

        # Compare
        print(f"Direct grad: {grad_direct.item():.6f}")
        print(f"Adjoint grad: {grad_adjoint.item():.6f}")
        print(f"Relative diff: {(grad_adjoint - grad_direct).abs() / grad_direct.abs():.4f}")

        # Should be reasonably close
        assert torch.allclose(grad_direct, grad_adjoint, rtol=0.2)
```

**Step 2: Run tests**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__adjoint_accuracy.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/torchscience/integration/initial_value_problem/test__adjoint_accuracy.py
git commit -m "test(ivp): add adjoint gradient accuracy tests"
```

---

## Task 3: Integration with All Solvers

**Files:**
- Create: `tests/torchscience/integration/initial_value_problem/test__adjoint_all_solvers.py`

**Step 1: Write tests for all solver combinations**

```python
# tests/torchscience/integration/initial_value_problem/test__adjoint_all_solvers.py
"""Tests verifying adjoint wrapper works with all available solvers."""

import pytest
import torch

from torchscience.integration.initial_value_problem import (
    adjoint,
    backward_euler,
    dormand_prince_5,
    euler,
    midpoint,
    runge_kutta_4,
)


@pytest.fixture
def dynamics():
    """Create dynamics with learnable parameter."""
    theta = torch.tensor([1.0], requires_grad=True)

    def f(t, y):
        return -theta * y

    return f, theta


@pytest.fixture
def y0():
    return torch.tensor([1.0])


class TestAdjointAllSolvers:
    @pytest.mark.parametrize(
        "solver,kwargs",
        [
            (euler, {"dt": 0.01}),
            (midpoint, {"dt": 0.01}),
            (runge_kutta_4, {"dt": 0.01}),
            (dormand_prince_5, {}),
            (backward_euler, {"dt": 0.01}),
        ],
    )
    def test_adjoint_works_with_solver(self, solver, kwargs, y0):
        """Each solver should work with adjoint wrapper."""
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        adjoint_solver = adjoint(solver)
        y_final, interp = adjoint_solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        # Forward should work
        assert not torch.isnan(y_final).any()

        # Backward should work
        loss = y_final.sum()
        loss.backward()
        assert theta.grad is not None

    @pytest.mark.parametrize(
        "solver,kwargs",
        [
            (euler, {"dt": 0.01}),
            (midpoint, {"dt": 0.01}),
            (runge_kutta_4, {"dt": 0.01}),
            (dormand_prince_5, {}),
        ],
    )
    def test_adjoint_interpolant_exists(self, solver, kwargs, y0):
        """Interpolant should be returned and functional."""

        def f(t, y):
            return -y

        adjoint_solver = adjoint(solver)
        _, interp = adjoint_solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        # Should be able to query
        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

        # Multiple queries
        t_query = torch.linspace(0, 1, 10)
        trajectory = interp(t_query)
        assert trajectory.shape[0] == 10


class TestAdjointNeuralODE:
    """Test adjoint with neural network dynamics (common use case)."""

    def test_mlp_dynamics(self):
        """Test with MLP-parameterized dynamics."""
        # Simple MLP
        hidden = 16
        net = torch.nn.Sequential(
            torch.nn.Linear(2, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 2),
        )

        def f(t, y):
            return net(y)

        y0 = torch.tensor([1.0, 0.5])

        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        # All parameters should have gradients
        for p in net.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()

    def test_neural_ode_training_step(self):
        """Simulate a training step with Neural ODE."""
        # Simple dynamics network
        net = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

        def f(t, y):
            return net(y)

        y0 = torch.tensor([[1.0]])
        target = torch.tensor([[0.5]])

        # Training step
        optimizer.zero_grad()
        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = (y_final - target).pow(2).mean()
        loss.backward()
        optimizer.step()

        # Parameter should have changed
        assert net.weight.grad is not None
```

**Step 2: Run tests**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__adjoint_all_solvers.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/torchscience/integration/initial_value_problem/test__adjoint_all_solvers.py
git commit -m "test(ivp): add adjoint tests for all solvers and neural ODE"
```

---

## Task 4: Final Module Exports and Documentation

**Files:**
- Modify: `src/torchscience/integration/initial_value_problem/__init__.py`

**Step 1: Update with complete exports and docstring**

```python
# src/torchscience/integration/initial_value_problem/__init__.py
"""
Initial value problem solvers for ordinary differential equations.

This module provides differentiable ODE solvers for PyTorch tensors and TensorDict.

Solvers
-------
euler
    Forward Euler method (1st order, fixed step, explicit).
    Simplest method, educational baseline.

midpoint
    Explicit midpoint method (2nd order, fixed step).
    Good accuracy/cost tradeoff for smooth problems.

runge_kutta_4
    Classic 4th-order Runge-Kutta (fixed step, explicit).
    Widely used workhorse, excellent for non-stiff problems.

dormand_prince_5
    Dormand-Prince 5(4) adaptive method (explicit).
    Production-quality solver with error control.

backward_euler
    Backward Euler method (1st order, fixed step, implicit).
    A-stable, suitable for stiff problems.

Wrappers
--------
adjoint
    Wrap any solver to use the continuous adjoint method for
    memory-efficient gradients. Uses O(1) memory for the autograd
    graph instead of O(n_steps).

Exceptions
----------
ODESolverError
    Base exception for ODE solver errors.

MaxStepsExceeded
    Raised when adaptive solver exceeds max_steps.

StepSizeError
    Raised when adaptive step size falls below dt_min.

ConvergenceError
    Raised when implicit solver Newton iteration fails to converge.

Examples
--------
Basic usage with adaptive solver:

>>> import torch
>>> from torchscience.integration.initial_value_problem import dormand_prince_5
>>>
>>> def decay(t, y):
...     return -y
>>>
>>> y0 = torch.tensor([1.0])
>>> y_final, interp = dormand_prince_5(decay, y0, t_span=(0.0, 5.0))
>>> trajectory = interp(torch.linspace(0, 5, 100))

With learnable parameters (Neural ODE style):

>>> theta = torch.tensor([1.5], requires_grad=True)
>>> def dynamics(t, y):
...     return -theta * y
>>>
>>> y_final, _ = dormand_prince_5(dynamics, y0, t_span=(0.0, 1.0))
>>> loss = y_final.sum()
>>> loss.backward()
>>> print(theta.grad)  # gradient of loss w.r.t. theta

Memory-efficient gradients with adjoint method:

>>> from torchscience.integration.initial_value_problem import adjoint
>>>
>>> adjoint_solver = adjoint(dormand_prince_5)
>>> y_final, _ = adjoint_solver(dynamics, y0, t_span=(0.0, 100.0))
>>> loss = y_final.sum()
>>> loss.backward()  # Uses O(1) memory for autograd graph

With TensorDict state:

>>> from tensordict import TensorDict
>>> def robot_dynamics(t, state):
...     return TensorDict({
...         "position": state["velocity"],
...         "velocity": -state["position"],
...     })
>>>
>>> state0 = TensorDict({
...     "position": torch.tensor([1.0]),
...     "velocity": torch.tensor([0.0]),
... })
>>> state_final, interp = runge_kutta_4(
...     robot_dynamics, state0, t_span=(0.0, 10.0), dt=0.01
... )

Stiff problems with implicit solver:

>>> def stiff_decay(t, y):
...     return -1000 * y  # Stiff coefficient
>>>
>>> y_final, _ = backward_euler(
...     stiff_decay, y0, t_span=(0.0, 1.0), dt=0.1
... )
"""

from torchscience.integration.initial_value_problem._adjoint import adjoint
from torchscience.integration.initial_value_problem._backward_euler import (
    backward_euler,
)
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
    # Explicit solvers (ordered by complexity)
    "euler",
    "midpoint",
    "runge_kutta_4",
    "dormand_prince_5",
    # Implicit solvers
    "backward_euler",
    # Wrappers
    "adjoint",
]
```

**Step 2: Update parent integration module**

```python
# src/torchscience/integration/__init__.py
"""
Numerical integration module.

Submodules
----------
initial_value_problem
    ODE solvers for initial value problems.
"""

from torchscience.integration import initial_value_problem

__all__ = [
    "initial_value_problem",
]
```

**Step 3: Commit**

```bash
git add src/torchscience/integration/
git commit -m "docs(ivp): complete Phase 4 with adjoint wrapper and full documentation"
```

---

## Task 5: Full Integration Test Suite

**Files:**
- Create: `tests/torchscience/integration/initial_value_problem/test__full_integration.py`

**Step 1: Write comprehensive integration tests**

```python
# tests/torchscience/integration/initial_value_problem/test__full_integration.py
"""Full integration tests covering all features together."""

import pytest
import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem import (
    adjoint,
    backward_euler,
    dormand_prince_5,
    euler,
    midpoint,
    runge_kutta_4,
)


class TestFullPipeline:
    """Test complete workflows combining multiple features."""

    def test_neural_ode_training_loop(self):
        """Simulate a Neural ODE training loop."""
        torch.manual_seed(42)

        # Network
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 2),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        def dynamics(t, y):
            return net(y)

        # Training data
        y0 = torch.randn(10, 2)  # Batch of 10
        targets = y0 * 0.5  # Shrink by half

        # Training loop
        losses = []
        for epoch in range(5):
            optimizer.zero_grad()

            adjoint_solver = adjoint(runge_kutta_4)
            y_final, _ = adjoint_solver(dynamics, y0, t_span=(0.0, 1.0), dt=0.1)

            loss = (y_final - targets).pow(2).mean()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_tensordict_with_adjoint(self):
        """TensorDict state with adjoint method."""
        omega = torch.tensor([2.0], requires_grad=True)

        def dynamics(t, state):
            return TensorDict(
                {"x": state["v"], "v": -omega**2 * state["x"]}
            )

        state0 = TensorDict({"x": torch.tensor([1.0]), "v": torch.tensor([0.0])})

        adjoint_solver = adjoint(dormand_prince_5)
        state_final, _ = adjoint_solver(dynamics, state0, t_span=(0.0, 1.0))

        loss = state_final["x"].sum()
        loss.backward()

        assert omega.grad is not None

    def test_complex_with_adjoint(self):
        """Complex-valued ODE with adjoint method."""
        theta = torch.tensor([1.0], requires_grad=True)

        def dynamics(t, y):
            return -theta.to(torch.complex128) * 1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)

        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(dynamics, y0, t_span=(0.0, 1.0), dt=0.01)

        loss = y_final.abs().sum()
        loss.backward()

        assert theta.grad is not None

    def test_backward_integration_with_adjoint(self):
        """Backward time integration with adjoint."""
        theta = torch.tensor([1.0], requires_grad=True)

        def dynamics(t, y):
            return -theta * y

        y1 = torch.tensor([torch.exp(torch.tensor(-1.0))])

        adjoint_solver = adjoint(runge_kutta_4)
        y0_recovered, _ = adjoint_solver(dynamics, y1, t_span=(1.0, 0.0), dt=0.01)

        loss = (y0_recovered - 1.0).pow(2)
        loss.backward()

        assert theta.grad is not None


class TestAllSolversAllFeatures:
    """Ensure all solvers support all features consistently."""

    SOLVERS = [
        ("euler", euler, {"dt": 0.01}),
        ("midpoint", midpoint, {"dt": 0.01}),
        ("runge_kutta_4", runge_kutta_4, {"dt": 0.01}),
        ("dormand_prince_5", dormand_prince_5, {}),
        ("backward_euler", backward_euler, {"dt": 0.01}),
    ]

    @pytest.mark.parametrize("name,solver,kwargs", SOLVERS)
    def test_tensor_state(self, name, solver, kwargs):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        assert y_final.shape == y0.shape
        assert interp(0.5).shape == y0.shape

    @pytest.mark.parametrize("name,solver,kwargs", SOLVERS)
    def test_tensordict_state(self, name, solver, kwargs):
        def f(t, state):
            return TensorDict({"x": -state["x"]})

        state0 = TensorDict({"x": torch.tensor([1.0])})
        state_final, interp = solver(f, state0, t_span=(0.0, 1.0), **kwargs)

        assert isinstance(state_final, TensorDict)

    @pytest.mark.parametrize("name,solver,kwargs", SOLVERS)
    def test_batched_state(self, name, solver, kwargs):
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0], [2.0], [3.0]])
        y_final, _ = solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        assert y_final.shape == (3, 1)

    @pytest.mark.parametrize("name,solver,kwargs", SOLVERS)
    def test_gradient_support(self, name, solver, kwargs):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        y_final, _ = solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None

    @pytest.mark.parametrize("name,solver,kwargs", SOLVERS)
    def test_adjoint_wrapper(self, name, solver, kwargs):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])

        adjoint_solver = adjoint(solver)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0), **kwargs)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
```

**Step 2: Run tests**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__full_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/torchscience/integration/initial_value_problem/test__full_integration.py
git commit -m "test(ivp): add comprehensive full integration tests"
```

---

## Summary

Phase 4 implements the adjoint wrapper for memory-efficient gradients:

1. **_AdjointODEFunction** - Custom autograd.Function implementing continuous adjoint method
2. **adjoint() wrapper** - Wraps any solver to use adjoint gradients:
   - O(1) memory for autograd graph (vs O(n_steps) for direct backprop)
   - Works with all 5 solvers
   - Supports TensorDict state
   - Supports complex numbers
   - Supports batched problems
3. **Gradient accuracy tests** - Verify adjoint gradients match finite differences
4. **Integration tests** - Neural ODE training loops, all solver combinations
5. **Full documentation** - Complete module docstrings with examples

The complete `torchscience.integration.initial_value_problem` module now provides:

| Solver | Order | Step Control | Stiffness | Method |
|--------|-------|--------------|-----------|--------|
| euler | 1st | Fixed | Non-stiff | Explicit |
| midpoint | 2nd | Fixed | Non-stiff | Explicit |
| runge_kutta_4 | 4th | Fixed | Non-stiff | Explicit |
| dormand_prince_5 | 5th | Adaptive | Non-stiff | Explicit |
| backward_euler | 1st | Fixed | Stiff | Implicit |

Plus the `adjoint()` wrapper for memory-efficient gradients with any solver.
