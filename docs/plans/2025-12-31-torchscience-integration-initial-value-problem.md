# Design: torchscience.integration.initial_value_problem

## Overview

The `torchscience.integration.initial_value_problem` module provides differentiable ODE solvers for PyTorch. It targets four use cases: physics-informed neural networks (PINNs), scientific simulation, control/robotics, and general scientific computing.

### Design Principles

- **Explicit over implicit** — Separate function per solver method, no unified dispatcher
- **Functional over object-oriented** — Returns tuples, not solution objects with methods
- **Simple by default** — Direct backprop, exceptions on failure, no stats clutter
- **Flexible when needed** — TensorDict for complex state, adjoint wrapper for memory efficiency

### Scope

- 5 ODE solvers spanning explicit, implicit, and adaptive methods
- Dense output via callable interpolant (differentiable)
- Full autograd support with optional adjoint method
- CPU and CUDA support with custom CUDA kernels
- Complex-valued states supported
- Batched integration intervals supported

### Out of Scope

- Quadrature (separate design)
- Boundary value problems (future)
- Partial differential equations (future)
- Event detection / root finding during integration (planned for future version)
- Symplectic integrators (future)

## Module Structure

```
torchscience/
  integration/
    __init__.py
    initial_value_problem/
      __init__.py
      _dormand_prince_5.py
      _euler.py
      _runge_kutta_4.py
      _backward_euler.py
      _midpoint.py
      _adjoint.py
  csrc/
    cpu/integration/
      initial_value_problem/
        dormand_prince_5.h
        euler.h
        runge_kutta_4.h
        backward_euler.h
        midpoint.h
    cuda/integration/
      initial_value_problem/
        dormand_prince_5.cu
        euler.cu
        runge_kutta_4.cu
        backward_euler.cu
        midpoint.cu
```

### Methods by Category

| Category | Method | Order | Step Control |
|----------|--------|-------|--------------|
| Explicit | `euler` | 1st | Fixed |
| Explicit | `midpoint` | 2nd | Fixed |
| Explicit | `runge_kutta_4` | 4th | Fixed |
| Explicit | `dormand_prince_5` | 5th | Adaptive |
| Implicit | `backward_euler` | 1st | Fixed |

### Public API

```python
from torchscience.integration.initial_value_problem import (
    # Solvers
    euler,
    midpoint,
    runge_kutta_4,
    dormand_prince_5,
    backward_euler,
    # Wrapper
    adjoint,
)
```

### Method Selection Guide

- `euler` — Educational baseline, simplest explicit method
- `midpoint` — Simple second-order, good accuracy/cost tradeoff
- `runge_kutta_4` — Classic workhorse, widely taught and used
- `dormand_prince_5` — Production adaptive solver, handles most non-stiff problems
- `backward_euler` — Simplest implicit method for stiff problems

## API Signatures

### Fixed-Step Solvers

```python
def euler(
    f,
    y0,
    t_span,
    dt,
    throw=True,
):
    """
    Solve ODE using forward Euler method.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
        Use closures or functools.partial to pass additional parameters.
    y0 : Tensor or TensorDict
        Initial state
    t_span : tuple[float, float] or tuple[Tensor, Tensor]
        Integration interval (t0, t1). Supports batched intervals.
    dt : float
        Fixed step size
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
```

Same signature for: `midpoint`, `runge_kutta_4`

### Implicit Solver

```python
def backward_euler(
    f,
    y0,
    t_span,
    dt,
    newton_tol=1e-6,
    max_newton_iter=10,
    throw=True,
):
    """
    Solve ODE using backward Euler (implicit) method.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
        Use closures or functools.partial to pass additional parameters.
    y0 : Tensor or TensorDict
        Initial state
    t_span : tuple[float, float] or tuple[Tensor, Tensor]
        Integration interval (t0, t1). Supports batched intervals.
    dt : float
        Fixed step size
    newton_tol : float
        Convergence tolerance for Newton iteration
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
```

### Adaptive Solver

```python
def dormand_prince_5(
    f,
    y0,
    t_span,
    rtol=1e-5,
    atol=1e-8,
    dt0=None,
    dt_min=None,
    dt_max=None,
    max_steps=10000,
    throw=True,
):
    """
    Solve ODE using Dormand-Prince 5(4) adaptive method.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
        Use closures or functools.partial to pass additional parameters.
    y0 : Tensor or TensorDict
        Initial state
    t_span : tuple[float, float] or tuple[Tensor, Tensor]
        Integration interval (t0, t1). Supports batched intervals.
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
```

### Dynamics Function Signature

```python
def f(t, y):
    """
    Parameters
    ----------
    t : float or Tensor
        Current time
    y : Tensor or TensorDict
        Current state

    Returns
    -------
    dydt : Tensor or TensorDict
        Time derivative, same structure as y

    Notes
    -----
    Use closures or functools.partial to pass additional parameters:

        # Closure approach
        def make_dynamics(omega, zeta):
            def f(t, y):
                return -omega * y - zeta * y**2
            return f

        # functools.partial approach
        def dynamics(t, y, omega, zeta):
            return -omega * y - zeta * y**2

        f = functools.partial(dynamics, omega=2.0, zeta=0.1)
    """
```

## Usage Examples

### Basic Usage with Tensor

```python
import torch
from torchscience.integration.initial_value_problem import (
    dormand_prince_5,
    euler,
)

# Simple exponential decay: dy/dt = -y
def decay(t, y):
    return -y

y0 = torch.tensor([1.0])
y_final, interp = dormand_prince_5(decay, y0, t_span=(0.0, 5.0))

# Query solution at specific times (differentiable)
t_query = torch.linspace(0, 5, 100)
trajectory = interp(t_query)  # shape: (100, 1)

# Backprop through the trajectory
loss = trajectory.sum()
loss.backward()  # works!
```

### With Parameters via Closures

```python
import functools

# Damped harmonic oscillator using closure
def make_oscillator(omega, zeta):
    def f(t, y):
        x, v = y[..., 0], y[..., 1]
        dxdt = v
        dvdt = -2 * zeta * omega * v - omega**2 * x
        return torch.stack([dxdt, dvdt], dim=-1)
    return f

y0 = torch.tensor([1.0, 0.0])  # [position, velocity]
y_final, interp = dormand_prince_5(
    make_oscillator(omega=2.0, zeta=0.1),
    y0,
    t_span=(0.0, 10.0),
)

# Alternative: functools.partial
def oscillator(t, y, omega, zeta):
    x, v = y[..., 0], y[..., 1]
    dxdt = v
    dvdt = -2 * zeta * omega * v - omega**2 * x
    return torch.stack([dxdt, dvdt], dim=-1)

y_final, interp = dormand_prince_5(
    functools.partial(oscillator, omega=2.0, zeta=0.1),
    y0,
    t_span=(0.0, 10.0),
)
```

### With TensorDict State

```python
from tensordict import TensorDict

def make_robot_dynamics(controller):
    def f(t, state):
        q = state["joints"]      # (7,) joint positions
        dq = state["velocities"] # (7,) joint velocities
        tau = controller(q, dq)  # control torques
        ddq = compute_forward_dynamics(q, dq, tau)
        return TensorDict({"joints": dq, "velocities": ddq})
    return f

state0 = TensorDict({
    "joints": torch.zeros(7),
    "velocities": torch.zeros(7),
})

state_final, interp = runge_kutta_4(
    make_robot_dynamics(my_controller),
    state0,
    t_span=(0.0, 10.0),
    dt=0.001,
)
# state_final is TensorDict with same structure
```

### With Autograd (PINNs Use Case)

```python
# Parameter we want to learn
theta = torch.tensor([1.5], requires_grad=True)

def dynamics(t, y):
    return -theta * y  # decay rate is learnable

y0 = torch.tensor([1.0])
y_final, _ = dormand_prince_5(dynamics, y0, t_span=(0.0, 1.0))

# Backprop through the solve
loss = (y_final - 0.5).pow(2)
loss.backward()
print(theta.grad)  # gradient of loss w.r.t. theta
```

### Batched Solves with Partial Failure Handling

```python
# When some batch elements may fail (stiff dynamics, etc.)
y0_batch = torch.randn(100, 2)  # 100 different initial conditions

y_final, interp = dormand_prince_5(
    stiff_dynamics,
    y0_batch,
    t_span=(0.0, 10.0),
    throw=False,  # don't raise on failure
)

# Check which elements succeeded
success = interp.success  # bool tensor, shape (100,)
print(f"{success.sum()} / 100 succeeded")

# Failed elements have NaN in y_final
valid_results = y_final[success]  # only successful results
```

### With Complex-Valued States

```python
# Schrodinger-like equation: dy/dt = -i * H * y
psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
H = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)

def schrodinger(t, psi):
    return -1j * H @ psi  # H captured via closure

psi_final, interp = dormand_prince_5(
    schrodinger,
    psi0,
    t_span=(0.0, 1.0),
)
```

## Adjoint Wrapper

### Default: Direct Backprop (Discretize-then-Optimize)

By default, all solvers store intermediate states and backpropagate through the solver steps directly. This gives exact gradients for the discretization but uses O(n_steps) memory.

### Memory-Efficient: Adjoint Method (Optimize-then-Discretize)

For large-scale problems, wrap any solver with `adjoint()` to use the adjoint method:

```python
from torchscience.integration.initial_value_problem import (
    dormand_prince_5,
    adjoint,
)

# Direct backprop (default) - O(n_steps) memory for autograd graph
y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 100.0))

# Adjoint method - O(1) memory for autograd graph
y_final, interp = adjoint(dormand_prince_5)(f, y0, t_span=(0.0, 100.0))
```

### How Adjoint Works

1. Forward pass solves the ODE normally, storing only checkpoints for gradient recomputation
2. Backward pass solves an augmented adjoint ODE backwards in time
3. Gradients w.r.t. parameters computed via integration, not backprop

### What Adjoint Affects (and Doesn't)

**Adjoint only affects gradient computation.** It does NOT change:

- The interpolant — still stores trajectory points for dense output
- Forward solve behavior — same numerical solution
- Return values — same `(y_final, interp)` tuple

**Memory breakdown:**

| Component | Direct Backprop | Adjoint |
|-----------|-----------------|---------|
| Autograd graph | O(n_steps) | O(1) |
| Interpolant storage | O(n_steps) | O(n_steps) |
| **Total** | O(n_steps) | O(n_steps) |

The memory savings come from not storing the autograd computation graph, not from discarding the trajectory. For problems where interpolant memory dominates (rare), consider:

- Not storing the interpolant (future `dense_output=False` option)
- Querying only `y_final` and discarding `interp`

### Adjoint Wrapper Signature

```python
def adjoint(solver, checkpoints=None):
    """
    Wrap a solver to use adjoint method for gradients.

    Parameters
    ----------
    solver : callable
        Any ODE solver function (euler, dormand_prince_5, etc.)
    checkpoints : int, optional
        Number of checkpoints for memory/compute tradeoff.
        None = automatic selection.

    Returns
    -------
    wrapped_solver : callable
        Solver with same signature but using adjoint gradients.
    """
```

### When to Use Adjoint

| Scenario | Recommendation |
|----------|----------------|
| Short integrations, small state | Direct (default) |
| Long integrations, large state | `adjoint()` |
| Need exact discretization gradients | Direct (default) |
| Memory-constrained | `adjoint()` |

## TensorDict Integration

TensorDict is a **hard dependency** of this module. It is always available and does not
require conditional imports or feature detection.

### Auto-Detection

Solvers automatically detect whether `y0` is a `Tensor` or `TensorDict` and preserve the type throughout:

```python
# Tensor in -> Tensor out
y0 = torch.tensor([1.0, 0.0])
y_final, interp = euler(f, y0, t_span, dt=0.01)
assert isinstance(y_final, torch.Tensor)
assert isinstance(interp(0.5), torch.Tensor)

# TensorDict in -> TensorDict out
y0 = TensorDict({"x": torch.zeros(3), "v": torch.ones(3)})
y_final, interp = euler(f, y0, t_span, dt=0.01)
assert isinstance(y_final, TensorDict)
assert isinstance(interp(0.5), TensorDict)
```

### Internal Handling

Solvers iterate over TensorDict leaves and concatenate into a flat vector for integration:

```python
# Inside each solver (conceptual)
if isinstance(y0, TensorDict):
    # Flatten TensorDict to vector
    leaves = [v.reshape(-1) for v in y0.values(flat_keys=True)]
    y_flat = torch.cat(leaves)

    # ... solve with flat tensor ...

    # Unflatten back to TensorDict structure
    y_final = _unflatten_to_tensordict(y_final_flat, y0)
```

### Nested TensorDict Support

Arbitrarily nested structures are supported:

```python
y0 = TensorDict({
    "robot": TensorDict({
        "joints": torch.zeros(7),
        "velocities": torch.zeros(7),
    }),
    "object": TensorDict({
        "pose": torch.zeros(6),
    }),
})
```

### Batch Dimensions

TensorDict batch dimensions are preserved. A state with batch shape `(B,)` produces outputs with batch shape `(B,)`.

## Dense Output and Interpolation

### Interpolant Contract

All solvers return an interpolant function as the second element of the tuple:

```python
y_final, interp = solver(f, y0, t_span, ...)

# Query single time
y_at_t = interp(5.0)  # Tensor or TensorDict

# Query multiple times
t_grid = torch.linspace(0, 10, 100)
trajectory = interp(t_grid)  # shape: (100, *state_shape)
```

### Output Shape Convention

**Time dimension is always first.** When querying multiple times:

```python
# y0 shape: (D,) — unbatched state
t_grid = torch.linspace(0, 10, 100)  # (T,)
trajectory = interp(t_grid)  # shape: (T, D) = (100, D)

# y0 shape: (B, D) — batched state
t_grid = torch.linspace(0, 10, 100)  # (T,)
trajectory = interp(t_grid)  # shape: (T, B, D) = (100, B, D)

# y0 shape: (B1, B2, D) — multi-batch state
t_grid = torch.linspace(0, 10, 100)  # (T,)
trajectory = interp(t_grid)  # shape: (T, B1, B2, D)
```

### Implementation by Method Type

| Method | Interpolation Strategy |
|--------|----------------------|
| Fixed-step (euler, rk4, etc.) | Store all steps, linear interpolation between |
| Adaptive (dormand_prince_5) | Hermite interpolation using embedded derivative info |

### Out-of-Bounds Behavior

```python
y_final, interp = solver(f, y0, t_span=(0.0, 10.0), ...)

interp(-1.0)   # Raises ValueError: t outside [0.0, 10.0]
interp(15.0)   # Raises ValueError: t outside [0.0, 10.0]
```

### Interpolant Differentiability

**Interpolants ARE differentiable.** Calling `interp(t)` returns a tensor that supports backpropagation:

```python
y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 10.0))

y_at_5 = interp(5.0)
y_at_5.requires_grad  # True
loss = y_at_5.sum()
loss.backward()       # Works!
```

This matches diffrax's approach where `sol.evaluate(t)` is differentiable. The gradients are computed
for the *numerical interpolation*, not an approximation to the derivative of the true ODE solution.

**Note on gradient accuracy:**

For the most accurate gradients at specific times, the numerical interpolation provides gradients for
the discretized solution. This is typically what users want for training neural ODEs or optimizing
parameters. The interpolant's mathematical derivative w.r.t. time (for computing `dy/dt`) may be
less accurate than re-evaluating the dynamics function directly.

### Batched Queries

```python
# Query at different times per batch element
t_batch = torch.tensor([1.0, 2.0, 3.0])  # (B,)
y_batch = interp(t_batch)  # (B, *state_shape)
```

### Batched t_span Semantics

When `t_span` contains tensors with batch dimensions:

```python
# Different integration intervals per batch element
t0 = torch.tensor([0.0, 0.0, 0.0])  # (B,)
t1 = torch.tensor([1.0, 2.0, 3.0])  # (B,) — different end times

y_final, interp = solver(f, y0, t_span=(t0, t1), dt=0.01)
```

**Behavior:**
- Fixed-step solvers use `dt` for all elements; elements reaching their `t1` early are held constant
- Adaptive solvers track per-element progress independently
- The interpolant's valid range is `[min(t0), max(t1)]` across all batch elements
- Querying outside an element's `[t0[i], t1[i]]` raises `ValueError`

### Backward Integration

Backward integration (integrating from larger to smaller time) is supported:

```python
# Integrate backwards from t=10 to t=0
y_final, interp = euler(f, y0, t_span=(10.0, 0.0), dt=0.1)

# dt is always positive; direction is inferred from t_span
# Internally: dt_actual = -dt when t1 < t0
```

The interpolant covers `[0.0, 10.0]` regardless of integration direction.

## Implementation Plan

### Phase 1: Core Infrastructure (via `dormand_prince_5`)

Implement the most complex solver first to establish all patterns:

1. Adaptive stepping — Step size control with embedded error estimation
2. Dense output — Hermite interpolation with derivative info (differentiable)
3. TensorDict handling — Using built-in flatten/unflatten
4. Autograd integration — Direct backprop through solver steps
5. Interpolant interface — Differentiable callable that queries arbitrary times
6. Complex dtype support
7. Batched t_span support
8. Custom CUDA kernel for fused stepping

### Phase 2: Fixed-Step Explicit Solvers

With infrastructure in place, implement simpler explicit methods in order of complexity:

1. `euler` — Validates fixed-step path, simplest explicit (1st order)
2. `midpoint` — Natural stepping stone (2nd order)
3. `runge_kutta_4` — Higher-order explicit, popular baseline (4th order)

Each solver includes both CPU and CUDA kernel implementations.

### Phase 3: Fixed-Step Implicit Solver

4. `backward_euler` — Implicit solver (requires internal Newton iteration)

### Phase 4: Adjoint Wrapper

5. `adjoint()` — Wraps any solver for memory-efficient gradients

### File Creation Order

```
Python API:
1. _dormand_prince_5.py    # Full adaptive solver (establishes patterns)
2. _euler.py               # Fixed-step explicit (1st order)
3. _midpoint.py            # Fixed-step explicit (2nd order)
4. _runge_kutta_4.py       # Fixed-step explicit (4th order)
5. _backward_euler.py      # Fixed-step implicit
6. _adjoint.py             # Gradient wrapper
7. __init__.py             # Public exports

C++/CUDA Kernels:
1. cpu/integration/initial_value_problem/dormand_prince_5.h
2. cuda/integration/initial_value_problem/dormand_prince_5.cu
3. cpu/integration/initial_value_problem/euler.h
4. cuda/integration/initial_value_problem/euler.cu
... (similar for other solvers)
```

### Testing Strategy

Each solver tested against:

- Analytical solutions (exponential decay, harmonic oscillator)
- SciPy reference implementations
- Gradient correctness via `torch.autograd.gradcheck`
- **Complex dtype support** (tested from day one):
  - Forward correctness with complex64/complex128 states
  - Gradient correctness via `torch.autograd.gradcheck` with complex states
  - Wirtinger derivative correctness for holomorphic and non-holomorphic dynamics
- TensorDict preservation
- Batched t_span correctness
- Backward integration correctness
- Interpolant differentiability tests
- `throw=False` partial failure tests (NaN output, success mask)
- `backward_euler`: Newton convergence tests
- CUDA kernel correctness (match CPU results)

## Exceptions

```python
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

## Design Decisions Summary

| Aspect | Decision |
|--------|----------|
| Module path | `torchscience.integration.initial_value_problem` |
| Methods | euler, midpoint, runge_kutta_4, dormand_prince_5, backward_euler |
| API style | Separate function per method |
| Return type | `(y, interp)` tuple |
| State types | Tensor or TensorDict (auto-detected) |
| TensorDict | Hard dependency (always available) |
| Complex dtypes | Fully supported and tested from day one |
| Batched t_span | Supported; elements reaching `t1` early are held constant |
| Backward integration | Supported; direction inferred from `t_span` |
| Dynamics signature | `f(t, y)` — use closures or `functools.partial` for parameters |
| Fixed-step param | `dt` |
| Adaptive params | `rtol=1e-5`, `atol=1e-8`, `dt0`, `dt_min`, `dt_max`, `max_steps` |
| Implicit params | `newton_tol`, `max_newton_iter` (Jacobian always via `torch.func.jacrev`) |
| `throw` param | `throw=True` default; when False, returns NaN for failures with `success` mask |
| Interpolant shape | Time dimension first: `(T, *state_shape)` |
| Interpolant differentiability | Differentiable; supports backpropagation (matches diffrax) |
| Gradients | Direct backprop default, `adjoint()` wrapper for memory-efficient |
| Adjoint memory | Saves autograd graph memory; interpolant storage unchanged |
| CUDA support | Custom CUDA kernels for all solvers |
| Errors | `MaxStepsExceeded`, `StepSizeError`, `ConvergenceError` (inherit from `ODESolverError`); only raised when `throw=True` |
| First implementation | `dormand_prince_5` |
| Implementation order | dormand_prince_5 → euler → midpoint → runge_kutta_4 → backward_euler → adjoint |

## Future Extensions (Out of Scope)

- Event detection (stop integration when condition met) — planned for future version
- Continuous callbacks (call function at each step)
- Stiff detection (auto-switch to implicit)
- Higher-order BDF methods
- Rosenbrock methods
- Exponential integrators
- Symplectic integrators (velocity_verlet)
