# Design: torchscience.integration.ordinary_differential_equation

## Overview

The `torchscience.integration.ordinary_differential_equation` module provides differentiable ODE solvers for PyTorch. It targets four use cases: physics-informed neural networks (PINNs), scientific simulation, control/robotics, and general scientific computing.

### Design Principles

- **Explicit over implicit** — Separate function per solver method, no unified dispatcher
- **Functional over object-oriented** — Returns tuples, not solution objects with methods
- **Simple by default** — Direct backprop, exceptions on failure, no stats clutter
- **Flexible when needed** — TensorDict for complex state, adjoint wrapper for memory efficiency

### Scope

- 7 ODE solvers spanning explicit, implicit, symplectic, and adaptive methods
- Dense output via callable interpolant
- Full autograd support with optional adjoint method
- CPU and CUDA support
- Complex-valued states supported
- Batched integration intervals supported

### Out of Scope

- Quadrature (separate design)
- Boundary value problems (future)
- Partial differential equations (future)
- Event detection / root finding during integration

## Module Structure

```
torchscience/
  integration/
    __init__.py
    ordinary_differential_equation/
      __init__.py
      _dormand_prince_5.py
      _euler.py
      _runge_kutta_4.py
      _backward_euler.py
      _leapfrog.py
      _verlet.py
      _midpoint.py
      _adjoint.py
```

### Methods by Category

| Category | Method | Order | Step Control |
|----------|--------|-------|--------------|
| Explicit | `euler` | 1st | Fixed |
| Explicit | `midpoint` | 2nd | Fixed |
| Explicit | `runge_kutta_4` | 4th | Fixed |
| Explicit | `dormand_prince_5` | 5th | Adaptive |
| Implicit | `backward_euler` | 1st | Fixed |
| Symplectic | `leapfrog` | 2nd | Fixed |
| Symplectic | `verlet` | 2nd | Fixed |

### Public API

```python
from torchscience.integration.ordinary_differential_equation import (
    # Solvers
    euler,
    midpoint,
    runge_kutta_4,
    dormand_prince_5,
    backward_euler,
    leapfrog,
    verlet,
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
- `leapfrog`, `verlet` — Energy-conserving methods for Hamiltonian systems

## API Signatures

### Fixed-Step Solvers

```python
def euler(
    f,
    y0,
    t_span,
    dt,
    func_args=(),
    func_kwargs=None,
):
    """
    Solve ODE using forward Euler method.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y, *args, **kwargs) -> dy/dt
    y0 : Tensor or TensorDict
        Initial state
    t_span : tuple[float, float] or tuple[Tensor, Tensor]
        Integration interval (t0, t1). Supports batched intervals.
    dt : float
        Fixed step size
    func_args : tuple
        Positional arguments passed to f
    func_kwargs : dict, optional
        Keyword arguments passed to f

    Returns
    -------
    y_final : Tensor or TensorDict
        State at t1, same type as y0
    interp : callable
        Interpolant function. interp(t) returns state at time(s) t.
    """
```

Same signature for: `midpoint`, `runge_kutta_4`, `backward_euler`, `leapfrog`, `verlet`

### Adaptive Solver

```python
def dormand_prince_5(
    f,
    y0,
    t_span,
    func_args=(),
    func_kwargs=None,
    rtol=1e-5,
    atol=1e-8,
    dt0=None,
    max_steps=10000,
):
    """
    Solve ODE using Dormand-Prince 5(4) adaptive method.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y, *args, **kwargs) -> dy/dt
    y0 : Tensor or TensorDict
        Initial state
    t_span : tuple[float, float] or tuple[Tensor, Tensor]
        Integration interval (t0, t1). Supports batched intervals.
    func_args : tuple
        Positional arguments passed to f
    func_kwargs : dict, optional
        Keyword arguments passed to f
    rtol : float
        Relative tolerance for step size control
    atol : float
        Absolute tolerance for step size control
    dt0 : float, optional
        Initial step size guess. If None, estimated automatically.
    max_steps : int
        Maximum number of steps before raising error

    Returns
    -------
    y_final : Tensor or TensorDict
        State at t1, same type as y0
    interp : callable
        Interpolant function. interp(t) returns state at time(s) t.
    """
```

### Dynamics Function Signature

```python
def f(t, y, *args, **kwargs):
    """
    Parameters
    ----------
    t : float or Tensor
        Current time
    y : Tensor or TensorDict
        Current state
    *args : tuple
        From func_args
    **kwargs : dict
        From func_kwargs

    Returns
    -------
    dydt : Tensor or TensorDict
        Time derivative, same structure as y
    """
```

## Usage Examples

### Basic Usage with Tensor

```python
import torch
from torchscience.integration.ordinary_differential_equation import (
    dormand_prince_5,
    euler,
)

# Simple exponential decay: dy/dt = -y
def decay(t, y):
    return -y

y0 = torch.tensor([1.0])
y_final, interp = dormand_prince_5(decay, y0, t_span=(0.0, 5.0))

# Query solution at specific times
t_eval = torch.linspace(0, 5, 100)
trajectory = interp(t_eval)  # shape: (100, 1)
```

### With Parameters via func_args

```python
# Damped harmonic oscillator
def oscillator(t, y, omega, zeta):
    x, v = y[..., 0], y[..., 1]
    dxdt = v
    dvdt = -2 * zeta * omega * v - omega**2 * x
    return torch.stack([dxdt, dvdt], dim=-1)

y0 = torch.tensor([1.0, 0.0])  # [position, velocity]
y_final, interp = dormand_prince_5(
    oscillator,
    y0,
    t_span=(0.0, 10.0),
    func_args=(2.0, 0.1),  # omega=2.0, zeta=0.1
)
```

### With TensorDict State

```python
from tensordict import TensorDict

def nbody(t, state, masses):
    pos = state["position"]  # (n_bodies, 3)
    vel = state["velocity"]  # (n_bodies, 3)
    acc = compute_gravitational_acceleration(pos, masses)
    return TensorDict({"position": vel, "velocity": acc})

state0 = TensorDict({
    "position": torch.randn(10, 3),
    "velocity": torch.randn(10, 3),
})
masses = torch.rand(10)

state_final, interp = verlet(
    nbody,
    state0,
    t_span=(0.0, 100.0),
    dt=0.01,
    func_args=(masses,),
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

### With Complex-Valued States

```python
# Schrodinger-like equation: dy/dt = -i * H * y
def schrodinger(t, psi, H):
    return -1j * H @ psi

psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
H = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)

psi_final, interp = dormand_prince_5(
    schrodinger,
    psi0,
    t_span=(0.0, 1.0),
    func_args=(H,),
)
```

## Adjoint Wrapper

### Default: Direct Backprop (Discretize-then-Optimize)

By default, all solvers store intermediate states and backpropagate through the solver steps directly. This gives exact gradients for the discretization but uses O(n_steps) memory.

### Memory-Efficient: Adjoint Method (Optimize-then-Discretize)

For large-scale problems, wrap any solver with `adjoint()` to use the adjoint method:

```python
from torchscience.integration.ordinary_differential_equation import (
    dormand_prince_5,
    adjoint,
)

# Direct backprop (default) - O(n_steps) memory
y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 100.0))

# Adjoint method - O(1) memory
y_final, interp = adjoint(dormand_prince_5)(f, y0, t_span=(0.0, 100.0))
```

### How Adjoint Works

1. Forward pass solves the ODE normally, storing only checkpoints
2. Backward pass solves an augmented adjoint ODE backwards in time
3. Gradients w.r.t. parameters computed via integration, not backprop

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

Uses TensorDict's built-in `flatten()` and `unflatten()` methods:

```python
# Inside each solver
if isinstance(y0, TensorDict):
    y_flat = y0.flatten()
    # ... solve with flat tensor ...
    y_final = y_final_flat.unflatten(y0.shape)
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

### Batched Queries

```python
# Query at different times per batch element
t_batch = torch.tensor([1.0, 2.0, 3.0])  # (B,)
y_batch = interp(t_batch)  # (B, *state_shape)
```

## Implementation Plan

### Phase 1: Core Infrastructure (via `dormand_prince_5`)

Implement the most complex solver first to establish all patterns:

1. Adaptive stepping — Step size control with embedded error estimation
2. Dense output — Hermite interpolation with derivative info
3. TensorDict handling — Using built-in flatten/unflatten
4. Autograd integration — Direct backprop through solver steps
5. Interpolant interface — Callable that queries arbitrary times
6. Complex dtype support
7. Batched t_span support

### Phase 2: Fixed-Step Solvers

With infrastructure in place, implement simpler methods:

1. `euler` — Validates fixed-step path, simplest explicit
2. `runge_kutta_4` — Higher-order explicit, popular baseline
3. `backward_euler` — Implicit solver (requires internal Newton iteration)

### Phase 3: Symplectic Solvers

4. `leapfrog` — Staggered velocity/position updates
5. `verlet` — Velocity Verlet variant

### Phase 4: Remaining Methods

6. `midpoint` — Second-order explicit

### Phase 5: Adjoint Wrapper

7. `adjoint()` — Wraps any solver for memory-efficient gradients

### File Creation Order

```
1. _dormand_prince_5.py    # Full adaptive solver (establishes patterns)
2. _euler.py               # Fixed-step explicit
3. _runge_kutta_4.py       # Fixed-step explicit
4. _backward_euler.py      # Fixed-step implicit
5. _leapfrog.py            # Symplectic
6. _verlet.py              # Symplectic
7. _midpoint.py            # Fixed-step explicit
8. _adjoint.py             # Gradient wrapper
9. __init__.py             # Public exports
```

### Testing Strategy

Each solver tested against:

- Analytical solutions (exponential decay, harmonic oscillator)
- SciPy reference implementations
- Gradient correctness via `torch.autograd.gradcheck`
- Complex-valued state correctness
- TensorDict preservation
- Batched t_span correctness

## Design Decisions Summary

| Aspect | Decision |
|--------|----------|
| Module path | `torchscience.integration.ordinary_differential_equation` |
| Methods | euler, midpoint, runge_kutta_4, dormand_prince_5, backward_euler, leapfrog, verlet |
| API style | Separate function per method |
| Return type | `(y_final, interp)` tuple |
| State types | Tensor or TensorDict (auto-detected) |
| Complex dtypes | Supported |
| Batched t_span | Supported |
| Dynamics signature | `f(t, y, *args, **kwargs)` |
| Solver params | `func_args=()`, `func_kwargs=None` |
| Fixed-step param | `dt` |
| Adaptive params | `rtol`, `atol`, `dt0`, `max_steps` |
| Gradients | Direct backprop default, `adjoint()` wrapper for memory-efficient |
| Errors | Exceptions only, no stats |
| First implementation | `dormand_prince_5` |

## Future Extensions (Out of Scope)

- Event detection (stop integration when condition met)
- Continuous callbacks (call function at each step)
- Stiff detection (auto-switch to implicit)
- Higher-order BDF methods
- Rosenbrock methods
- Exponential integrators
