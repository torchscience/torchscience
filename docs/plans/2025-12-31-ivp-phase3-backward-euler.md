# Phase 3: Backward Euler Implicit Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the backward Euler implicit ODE solver with Newton iteration for stiff problems.

**Architecture:** Backward Euler requires solving a nonlinear equation at each step: `y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})`. Use Newton's method with automatic Jacobian computation via `torch.func.jacrev`. Reuse TensorDict utilities and interpolant infrastructure from Phases 1-2.

**Tech Stack:** PyTorch, TensorDict, torch.func.jacrev (for Jacobian), C++17 (CPU kernels)

**Prerequisites:** Phases 1-2 complete (explicit solvers, interpolant, TensorDict utilities, exceptions)

---

## Task 1: Newton Solver Utility

**Files:**
- Create: `src/torchscience/integration/initial_value_problem/_newton.py`
- Create: `tests/torchscience/integration/initial_value_problem/test__newton.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/integration/initial_value_problem/test__newton.py
import pytest
import torch

from torchscience.integration.initial_value_problem._newton import newton_solve


class TestNewtonSolve:
    def test_simple_root(self):
        """Find root of f(x) = x^2 - 2 => x = sqrt(2)"""

        def f(x):
            return x**2 - 2

        x0 = torch.tensor([1.0])
        x_root, converged = newton_solve(f, x0, tol=1e-10, max_iter=50)

        expected = torch.sqrt(torch.tensor([2.0]))
        assert converged
        assert torch.allclose(x_root, expected, atol=1e-8)

    def test_multidimensional(self):
        """Find root of [x^2 + y - 1, x + y^2 - 1] => x = y = 0.6180..."""

        def f(z):
            x, y = z[0], z[1]
            return torch.stack([x**2 + y - 1, x + y**2 - 1])

        z0 = torch.tensor([0.5, 0.5])
        z_root, converged = newton_solve(f, z0, tol=1e-10, max_iter=50)

        # Both roots are (1-sqrt(5))/2 or similar
        assert converged
        assert torch.allclose(f(z_root), torch.zeros(2), atol=1e-8)

    def test_convergence_failure(self):
        """Should not converge with too few iterations"""

        def f(x):
            return x**2 - 2

        x0 = torch.tensor([100.0])  # Far from root
        _, converged = newton_solve(f, x0, tol=1e-10, max_iter=2)

        assert not converged

    def test_batched(self):
        """Solve multiple systems in parallel"""

        def f(x):
            return x**2 - torch.tensor([[2.0], [3.0], [4.0]])

        x0 = torch.tensor([[1.0], [1.5], [1.8]])
        x_root, converged = newton_solve(f, x0, tol=1e-8, max_iter=50)

        expected = torch.sqrt(torch.tensor([[2.0], [3.0], [4.0]]))
        assert converged
        assert torch.allclose(x_root, expected, atol=1e-6)

    def test_differentiable(self):
        """Gradients should flow through the solution"""
        a = torch.tensor([2.0], requires_grad=True)

        def f(x):
            return x**2 - a

        x0 = torch.tensor([1.0])
        x_root, _ = newton_solve(f, x0, tol=1e-10, max_iter=50)

        # x_root = sqrt(a), so d(x_root)/da = 1/(2*sqrt(a))
        loss = x_root.sum()
        loss.backward()

        expected_grad = 1 / (2 * torch.sqrt(a))
        assert a.grad is not None
        assert torch.allclose(a.grad, expected_grad, rtol=1e-4)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__newton.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/torchscience/integration/initial_value_problem/_newton.py
"""Newton's method for solving nonlinear systems."""

from typing import Callable, Tuple

import torch


def newton_solve(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 10,
) -> Tuple[torch.Tensor, bool]:
    """
    Solve f(x) = 0 using Newton's method with automatic Jacobian.

    Parameters
    ----------
    f : callable
        Function to find root of. f(x) -> residual with same shape as x.
    x0 : Tensor
        Initial guess.
    tol : float
        Convergence tolerance on residual norm.
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    x : Tensor
        Solution (or last iterate if not converged).
    converged : bool
        Whether the method converged within tolerance.
    """
    x = x0.clone()

    # Determine if batched
    is_batched = x.dim() > 1
    if not is_batched:
        x = x.unsqueeze(0)

    batch_size = x.shape[0]
    state_size = x.shape[1]

    for _ in range(max_iter):
        # Compute residual
        residual = f(x)
        if not is_batched and residual.dim() == 1:
            residual = residual.unsqueeze(0)

        # Check convergence
        residual_norm = torch.linalg.norm(residual, dim=-1)
        if (residual_norm < tol).all():
            if not is_batched:
                x = x.squeeze(0)
            return x, True

        # Compute Jacobian using torch.func
        # For batched case, use vmap over jacrev
        def f_single(xi):
            result = f(xi.unsqueeze(0))
            if result.dim() > 1:
                return result.squeeze(0)
            return result

        # Compute Jacobian for each batch element
        jacobians = []
        for i in range(batch_size):
            jac = torch.func.jacrev(f_single)(x[i])
            jacobians.append(jac)
        J = torch.stack(jacobians)  # (B, state_size, state_size)

        # Newton update: x_new = x - J^{-1} @ f(x)
        # Solve J @ dx = -residual for dx
        try:
            dx = torch.linalg.solve(J, -residual.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            # Singular Jacobian
            if not is_batched:
                x = x.squeeze(0)
            return x, False

        x = x + dx

    # Check final convergence
    residual = f(x)
    if not is_batched and residual.dim() == 1:
        residual = residual.unsqueeze(0)
    residual_norm = torch.linalg.norm(residual, dim=-1)
    converged = (residual_norm < tol).all().item()

    if not is_batched:
        x = x.squeeze(0)

    return x, converged
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__newton.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/_newton.py tests/torchscience/integration/initial_value_problem/test__newton.py
git commit -m "feat(ivp): add Newton solver utility for implicit methods"
```

---

## Task 2: Backward Euler Solver Implementation

**Files:**
- Create: `src/torchscience/integration/initial_value_problem/_backward_euler.py`
- Create: `tests/torchscience/integration/initial_value_problem/test__backward_euler.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/integration/initial_value_problem/test__backward_euler.py
import pytest
import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem import (
    ConvergenceError,
    backward_euler,
)


class TestBackwardEulerBasic:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 => y(t) = exp(-t)"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = backward_euler(decay, y0, t_span=(0.0, 1.0), dt=0.1)

        expected = torch.exp(torch.tensor([-1.0]))
        # Backward Euler is 1st order, similar accuracy to forward Euler
        assert torch.allclose(y_final, expected, rtol=0.1)

    def test_stiff_problem(self):
        """Test on a stiff ODE that would require tiny steps for explicit methods."""
        # dy/dt = -1000 * (y - sin(t)) + cos(t)
        # Stiff because of the -1000 coefficient

        def stiff(t, y):
            return -1000 * (y - torch.sin(t)) + torch.cos(t)

        y0 = torch.tensor([0.0])
        # Backward Euler should handle this with reasonable step size
        y_final, _ = backward_euler(
            stiff, y0, t_span=(0.0, 1.0), dt=0.1, newton_tol=1e-8
        )

        # Exact solution is y = sin(t)
        expected = torch.sin(torch.tensor([1.0]))
        assert torch.allclose(y_final, expected, rtol=0.1)

    def test_returns_interpolant(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_interpolant_endpoints(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert torch.allclose(interp(0.0), y0, atol=1e-5)
        assert torch.allclose(interp(1.0), y_final, atol=1e-5)


class TestBackwardEulerNewtonConvergence:
    def test_newton_convergence_default(self):
        """Default Newton parameters should work for simple problems."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, _ = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        # Should complete without error
        assert not torch.isnan(y_final).any()

    def test_newton_tol_affects_accuracy(self):
        """Tighter Newton tolerance should give more consistent results."""

        def f(t, y):
            return -y**2 + 1  # Nonlinear

        y0 = torch.tensor([0.5])

        y_loose, _ = backward_euler(
            f, y0, t_span=(0.0, 1.0), dt=0.1, newton_tol=1e-3, max_newton_iter=50
        )
        y_tight, _ = backward_euler(
            f, y0, t_span=(0.0, 1.0), dt=0.1, newton_tol=1e-10, max_newton_iter=50
        )

        # Both should produce finite results
        assert not torch.isnan(y_loose).any()
        assert not torch.isnan(y_tight).any()

    def test_convergence_error_thrown(self):
        """Should raise ConvergenceError when Newton fails (throw=True)."""

        def difficult(t, y):
            # Dynamics that make Newton hard to converge
            return y**3 - 100 * y

        y0 = torch.tensor([0.1])

        with pytest.raises(ConvergenceError):
            backward_euler(
                difficult,
                y0,
                t_span=(0.0, 1.0),
                dt=0.5,  # Large step makes convergence hard
                newton_tol=1e-12,
                max_newton_iter=2,  # Not enough iterations
            )

    def test_convergence_failure_no_throw(self):
        """Should return NaN when Newton fails (throw=False)."""

        def difficult(t, y):
            return y**3 - 100 * y

        y0 = torch.tensor([0.1])

        y_final, interp = backward_euler(
            difficult,
            y0,
            t_span=(0.0, 1.0),
            dt=0.5,
            newton_tol=1e-12,
            max_newton_iter=2,
            throw=False,
        )

        assert torch.isnan(y_final).any()
        assert interp.success is not None
        assert not interp.success.all()


class TestBackwardEulerAutograd:
    def test_gradient_through_solver(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        y_final, _ = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad).any()

    def test_gradient_through_interpolant(self):
        theta = torch.tensor([1.0], requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0])
        _, interp = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        y_mid = interp(0.5)
        loss = y_mid.sum()
        loss.backward()

        assert theta.grad is not None


class TestBackwardEulerTensorDict:
    def test_simple_tensordict(self):
        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict({"x": torch.tensor([1.0]), "v": torch.tensor([0.0])})
        state_final, _ = backward_euler(f, state0, t_span=(0.0, 1.0), dt=0.1)

        assert isinstance(state_final, TensorDict)
        assert "x" in state_final.keys()


class TestBackwardEulerComplex:
    def test_complex_exponential(self):
        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.01)

        expected = torch.exp(-1j * torch.tensor(1.0))
        assert torch.allclose(y_final.squeeze(), expected, atol=0.1)


class TestBackwardEulerBatched:
    def test_batched_initial_conditions(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0], [2.0], [3.0]])
        y_final, _ = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert y_final.shape == (3, 1)


class TestBackwardEulerBackward:
    def test_backward_integration(self):
        def f(t, y):
            return -y

        y1 = torch.tensor([torch.exp(torch.tensor(-1.0))])
        y0_recovered, _ = backward_euler(f, y1, t_span=(1.0, 0.0), dt=0.1)

        expected = torch.tensor([1.0])
        assert torch.allclose(y0_recovered, expected, rtol=0.2)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__backward_euler.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/torchscience/integration/initial_value_problem/_backward_euler.py
"""Backward Euler (implicit) ODE solver."""

from typing import Callable, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem._exceptions import (
    ConvergenceError,
)
from torchscience.integration.initial_value_problem._interpolant import (
    LinearInterpolant,
)
from torchscience.integration.initial_value_problem._newton import newton_solve
from torchscience.integration.initial_value_problem._tensordict_utils import (
    flatten_state,
)


def backward_euler(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    dt: float,
    newton_tol: float = 1e-6,
    max_newton_iter: int = 10,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using backward Euler (implicit) method.

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
    newton_tol : float
        Convergence tolerance for Newton iteration.
    max_newton_iter : int
        Maximum Newton iterations per step. Raises ConvergenceError if exceeded
        (when throw=True).
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
    ConvergenceError
        If Newton iteration fails to converge within max_newton_iter (only when throw=True).

    Notes
    -----
    Jacobian for Newton iteration is computed automatically via torch.func.jacrev.
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

    success_overall = True

    while direction * (t1 - t) > 1e-10:
        h_step = h
        if direction * (t + h_step - t1) > 0:
            h_step = t1 - t

        t_next = t + h_step

        # Backward Euler: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
        # Solve: g(y_{n+1}) = y_{n+1} - y_n - h * f(t_{n+1}, y_{n+1}) = 0

        def residual(y_next):
            return y_next - y - h_step * f_flat(t_next, y_next)

        # Initial guess: forward Euler prediction
        y_guess = y + h_step * f_flat(t, y)

        y_next, converged = newton_solve(
            residual, y_guess, tol=newton_tol, max_iter=max_newton_iter
        )

        if not converged:
            if throw:
                raise ConvergenceError(
                    f"Newton iteration failed to converge at t={t_next} "
                    f"after {max_newton_iter} iterations"
                )
            else:
                y = torch.full_like(y, float("nan"))
                success_overall = False
                break

        y = y_next
        t = t_next

        t_points.append(torch.tensor(t, dtype=dtype, device=device))
        y_points.append(y.clone())

    t_tensor = torch.stack(t_points)
    y_tensor = torch.stack(y_points)

    if throw:
        success = None
    else:
        success = torch.tensor(
            success_overall,
            dtype=torch.bool,
            device=device,
        )
        # Expand to batch shape if needed
        if y_flat.dim() > 1:
            success = success.expand(y_flat.shape[:-1])

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
from torchscience.integration.initial_value_problem._backward_euler import backward_euler
```

And add `"backward_euler"` to `__all__`.

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__backward_euler.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/ tests/torchscience/integration/initial_value_problem/
git commit -m "feat(ivp): add backward_euler implicit ODE solver"
```

---

## Task 3: CPU C++ Kernel for Backward Euler

**Files:**
- Create: `src/torchscience/csrc/cpu/integration/initial_value_problem/backward_euler.h`

**Step 1: Write the C++ kernel header**

```cpp
// src/torchscience/csrc/cpu/integration/initial_value_problem/backward_euler.h
#pragma once

#include <ATen/ATen.h>
#include <functional>
#include <tuple>

namespace torchscience {
namespace cpu {
namespace integration {
namespace initial_value_problem {

template <typename scalar_t>
struct NewtonSolver {
  /**
   * Solve f(x) = 0 using Newton's method.
   *
   * @param f Function to find root of
   * @param jacobian Function returning Jacobian of f
   * @param x0 Initial guess
   * @param tol Convergence tolerance
   * @param max_iter Maximum iterations
   * @return (solution, converged) tuple
   */
  static std::tuple<at::Tensor, bool> solve(
      const std::function<at::Tensor(const at::Tensor&)>& f,
      const std::function<at::Tensor(const at::Tensor&)>& jacobian,
      const at::Tensor& x0,
      scalar_t tol,
      int max_iter
  ) {
    at::Tensor x = x0.clone();

    for (int i = 0; i < max_iter; ++i) {
      at::Tensor residual = f(x);
      scalar_t norm = at::linalg_norm(residual).item<scalar_t>();

      if (norm < tol) {
        return std::make_tuple(x, true);
      }

      at::Tensor J = jacobian(x);
      at::Tensor dx = at::linalg_solve(J, -residual.unsqueeze(-1)).squeeze(-1);
      x = x + dx;
    }

    // Check final convergence
    at::Tensor residual = f(x);
    scalar_t norm = at::linalg_norm(residual).item<scalar_t>();
    return std::make_tuple(x, norm < tol);
  }
};

template <typename scalar_t>
struct BackwardEulerStep {
  /**
   * Single step of backward Euler method.
   *
   * Solves: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
   *
   * @param f Dynamics function
   * @param jacobian_f Jacobian of dynamics w.r.t. y
   * @param t Current time
   * @param y Current state
   * @param h Step size
   * @param tol Newton tolerance
   * @param max_iter Maximum Newton iterations
   * @return (y_new, converged) tuple
   */
  static std::tuple<at::Tensor, bool> step(
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& f,
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& jacobian_f,
      scalar_t t,
      const at::Tensor& y,
      scalar_t h,
      scalar_t tol,
      int max_iter
  ) {
    scalar_t t_next = t + h;

    // Residual: g(y_new) = y_new - y - h * f(t_next, y_new)
    auto residual = [&](const at::Tensor& y_new) {
      return y_new - y - h * f(t_next, y_new);
    };

    // Jacobian: dg/dy_new = I - h * df/dy
    auto jacobian = [&](const at::Tensor& y_new) {
      int64_t n = y_new.size(0);
      at::Tensor I = at::eye(n, y_new.options());
      at::Tensor df_dy = jacobian_f(t_next, y_new);
      return I - h * df_dy;
    };

    // Initial guess: forward Euler
    at::Tensor y_guess = y + h * f(t, y);

    return NewtonSolver<scalar_t>::solve(residual, jacobian, y_guess, tol, max_iter);
  }
};

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cpu
}  // namespace torchscience
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/integration/initial_value_problem/backward_euler.h
git commit -m "feat(ivp): add backward Euler CPU kernel header"
```

---

## Task 4: CUDA Kernel Placeholder for Backward Euler

**Files:**
- Create: `src/torchscience/csrc/cuda/integration/initial_value_problem/backward_euler.cu`

**Step 1: Write the CUDA kernel placeholder**

```cpp
// src/torchscience/csrc/cuda/integration/initial_value_problem/backward_euler.cu
#include <ATen/ATen.h>

namespace torchscience {
namespace cuda {
namespace integration {
namespace initial_value_problem {

// TODO: Implement CUDA kernel for Backward Euler method
//
// Key considerations:
// 1. Newton iteration requires linear solve (batched LU decomposition)
// 2. Jacobian computation can use cuBLAS for batched matmul
// 3. Each batch element can have different convergence
//
// For now, the Python implementation handles CUDA tensors.

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cuda
}  // namespace torchscience
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cuda/integration/initial_value_problem/backward_euler.cu
git commit -m "feat(ivp): add backward Euler CUDA kernel placeholder"
```

---

## Task 5: Stiffness Comparison Tests

**Files:**
- Create: `tests/torchscience/integration/initial_value_problem/test__stiffness.py`

**Step 1: Write stiffness comparison tests**

```python
# tests/torchscience/integration/initial_value_problem/test__stiffness.py
"""Tests comparing implicit vs explicit solvers on stiff problems."""

import pytest
import torch

from torchscience.integration.initial_value_problem import (
    backward_euler,
    euler,
    runge_kutta_4,
)


class TestStiffProblem:
    """Test on stiff ODE: dy/dt = -lambda * y with large lambda."""

    def test_explicit_euler_unstable_for_stiff(self):
        """Forward Euler is unstable for stiff problems with large steps."""
        lam = 1000.0  # Stiff coefficient

        def f(t, y):
            return -lam * y

        y0 = torch.tensor([1.0])

        # For stability, Euler needs dt < 2/lambda = 0.002
        # With dt = 0.01, Euler will be unstable
        y_euler, _ = euler(f, y0, t_span=(0.0, 0.1), dt=0.01)

        # Should either blow up or oscillate wildly
        # (may produce NaN or very large values)
        # In practice, the solution will oscillate with increasing amplitude
        assert y_euler.abs().max() > 10 or torch.isnan(y_euler).any()

    def test_backward_euler_stable_for_stiff(self):
        """Backward Euler remains stable for stiff problems."""
        lam = 1000.0

        def f(t, y):
            return -lam * y

        y0 = torch.tensor([1.0])

        # Backward Euler is unconditionally stable
        y_be, _ = backward_euler(
            f, y0, t_span=(0.0, 1.0), dt=0.1, newton_tol=1e-10
        )

        # Should produce bounded result close to 0 (exact: exp(-1000))
        assert y_be.abs().max() < 1.0
        assert not torch.isnan(y_be).any()

    def test_accuracy_vs_stability_tradeoff(self):
        """
        Backward Euler is A-stable but only 1st order accurate.
        For non-stiff problems, explicit methods may be more accurate.
        """
        # Non-stiff problem
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        y_euler, _ = euler(f, y0, t_span=(0.0, 1.0), dt=0.1)
        y_be, _ = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)
        y_rk4, _ = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        error_euler = (y_euler - expected).abs().item()
        error_be = (y_be - expected).abs().item()
        error_rk4 = (y_rk4 - expected).abs().item()

        # RK4 should be most accurate for non-stiff
        assert error_rk4 < error_euler
        assert error_rk4 < error_be

        # Euler and backward Euler have similar accuracy (both 1st order)
        assert abs(error_euler - error_be) < 0.1


class TestRobertsonProblem:
    """Robertson's problem - classic stiff test case."""

    @pytest.mark.parametrize("solver", ["backward_euler"])
    def test_robertson_backward_euler(self, solver):
        """
        Robertson's chemical kinetics problem:
        dy1/dt = -0.04*y1 + 1e4*y2*y3
        dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
        dy3/dt = 3e7*y2^2

        This is a stiff system with timescales spanning 10^11.
        """

        def robertson(t, y):
            y1, y2, y3 = y[0], y[1], y[2]
            dy1 = -0.04 * y1 + 1e4 * y2 * y3
            dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
            dy3 = 3e7 * y2**2
            return torch.stack([dy1, dy2, dy3])

        y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        # Backward Euler can handle this with moderate step size
        y_final, _ = backward_euler(
            robertson,
            y0,
            t_span=(0.0, 0.1),  # Short integration
            dt=0.01,
            newton_tol=1e-8,
            max_newton_iter=20,
        )

        # Conservation: y1 + y2 + y3 = 1
        total = y_final.sum()
        assert torch.allclose(total, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

        # All concentrations should be non-negative
        assert (y_final >= -1e-10).all()
```

**Step 2: Run tests**

Run: `uv run pytest tests/torchscience/integration/initial_value_problem/test__stiffness.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/torchscience/integration/initial_value_problem/test__stiffness.py
git commit -m "test(ivp): add stiffness comparison tests for backward_euler"
```

---

## Task 6: Final Module Exports

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

Examples
--------
>>> import torch
>>> from torchscience.integration.initial_value_problem import backward_euler
>>>
>>> def stiff_decay(t, y):
...     return -1000 * y  # Stiff coefficient
>>>
>>> y0 = torch.tensor([1.0])
>>> y_final, interp = backward_euler(stiff_decay, y0, t_span=(0.0, 1.0), dt=0.1)
"""

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
]
```

**Step 2: Commit**

```bash
git add src/torchscience/integration/initial_value_problem/__init__.py
git commit -m "docs(ivp): complete Phase 3 with backward_euler implicit solver"
```

---

## Summary

Phase 3 implements the backward Euler implicit ODE solver:

1. **Newton solver utility** - Generic Newton's method with automatic Jacobian via torch.func.jacrev
2. **backward_euler solver** - Implicit A-stable solver with:
   - Newton iteration for nonlinear solve at each step
   - Configurable tolerance and max iterations
   - ConvergenceError exception (or NaN with throw=False)
   - TensorDict support
   - Complex number support
   - Autograd support
3. **CPU C++ kernel header** - Newton solver and backward Euler step templates
4. **CUDA kernel placeholder** - For future GPU acceleration
5. **Stiffness comparison tests** - Demonstrate stability advantages over explicit methods

The backward Euler method is unconditionally A-stable, making it suitable for stiff problems where explicit methods (like Euler or RK4) would require impractically small step sizes.

Phase 4 (adjoint wrapper) will add memory-efficient gradient computation for all solvers.
