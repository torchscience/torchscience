# IVP Tutorial Notebook Design

## Overview

Create a showcase Jupyter notebook demonstrating `torchscience.integration.initial_value_problem` features. The notebook should be visually impressive (marketing-quality) while also being educational for researchers, ML practitioners, and SciPy users migrating to PyTorch.

## File Location

```
notebooks/torchscience/integration/initial_value_problem.ipynb
```

Mirrors the package structure for discoverability.

## Dependencies

- `torchscience`
- `plotly` - interactive visualizations
- `manim` (community edition) - cinematic animations via `%%manim` magic
- `IPython` - video display

## Visualization Strategy

**Hybrid approach:**
- **Plotly** for interactive exploration (rotate 3D trajectories, zoom phase portraits)
- **Manim** for hero animations (pendulum swinging, populations evolving, Neural ODE flow)

All Manim animations render inline using `%%manim` cell magic.

## Notebook Structure (~35 cells)

### Section 1: Introduction (3 cells)

| Cell | Type | Content |
|------|------|---------|
| 1 | Markdown | Title, overview of IVP solvers, what we'll build |
| 2 | Code | Imports: torch, torchscience, plotly, manim |
| 3 | Code | Setup: device selection, random seed, plotting defaults |

### Section 2: Double Pendulum (10 cells)

**Physics:** Two masses connected by rigid rods swinging under gravity. State vector `[theta1, theta2, omega1, omega2]`. Chaotic dynamics showcase adaptive stepping.

| Cell | Type | Content |
|------|------|---------|
| 1 | Markdown | Brief intro to double pendulum, chaos, why it's a good test case |
| 2 | Code | Define `double_pendulum_dynamics(t, state)` - the ODE RHS |
| 3 | Code | Set initial conditions (slightly off vertical for dramatic chaos) |
| 4 | Code | Solve with `dormand_prince_5`, show timing and step count |
| 5 | Code | Use interpolant to get dense trajectory at 1000 time points |
| 6 | Code | Plotly 3D plot: (theta1, theta2, t) showing chaotic trajectory |
| 7 | Code | Plotly 2D phase portrait: (theta1, omega1) colored by time |
| 8 | Code | Convert angles to (x, y) Cartesian coordinates for both masses |
| 9 | `%%manim` | `DoublePendulumScene` - animate pendulum swinging with trailing path |
| 10 | Markdown | Observations: adaptive stepping, sensitivity to initial conditions |

**Manim animation:** Two-armed pendulum swinging with fading trail showing recent trajectory. Optionally show second pendulum with perturbed initial conditions to demonstrate chaos (divergence).

**Features demonstrated:**
- `dormand_prince_5` adaptive solver
- Dense output via interpolant
- Step size adaptation in chaotic regions

### Section 3: Lotka-Volterra Predator-Prey (10 cells)

**Physics:** Classic ecology model - prey (rabbits) and predators (foxes). Populations oscillate in closed orbits. Uses TensorDict for structured state.

| Cell | Type | Content |
|------|------|---------|
| 1 | Markdown | Ecology intro, Lotka-Volterra equations, real-world context |
| 2 | Code | Define `lotka_volterra(t, state)` returning TensorDict |
| 3 | Code | Create TensorDict initial state, set parameters (alpha, beta, delta, gamma) |
| 4 | Code | Solve with `runge_kutta_4` - demonstrate fixed-step with TensorDict |
| 5 | Code | Solver comparison: `euler`, `midpoint`, `runge_kutta_4`, `dormand_prince_5` |
| 6 | Code | Plotly: population time series (prey and predator vs time) |
| 7 | Code | Plotly: interactive phase portrait with multiple initial conditions |
| 8 | `%%manim` | `LotkaVolterraScene` - split view animation |
| 9 | Markdown | Discussion: conservation laws, closed orbits, solver selection |
| 10 | Code | Parameter sensitivity - change alpha, show orbit changes |

**Manim animation:** Two-panel view. Left: stylized rabbit and fox icons growing/shrinking with population. Right: phase portrait being traced in real-time, showing closed orbit forming.

**Features demonstrated:**
- TensorDict state support
- Solver comparison (accuracy vs cost)
- Fixed-step solvers (`euler`, `midpoint`, `runge_kutta_4`)
- Parameter sensitivity

### Section 4: Neural ODE Spiral Classification (10 cells)

**ML setup:** Classify points from two interleaved spirals using continuous-depth neural network. Demonstrates differentiability and adjoint method for memory-efficient gradients.

| Cell | Type | Content |
|------|------|---------|
| 1 | Markdown | Neural ODE concept, continuous depth, adjoint memory benefits |
| 2 | Code | Generate spiral dataset (two interleaved spirals) |
| 3 | Code | Plotly: visualize raw spiral data |
| 4 | Code | Define `ODEFunc(nn.Module)` - small MLP for dynamics |
| 5 | Code | Define `NeuralODE` model: encoder -> integrate -> classifier |
| 6 | Code | Training loop with `adjoint(dormand_prince_5)`, show loss curve |
| 7 | Code | Memory comparison: regular solver vs adjoint-wrapped |
| 8 | Code | Plotly: final decision boundary overlaid on spiral data |
| 9 | `%%manim` | `NeuralODEScene` - hidden states flowing, boundary evolving |
| 10 | Markdown | Key takeaways: when to use adjoint, gradient accuracy tradeoffs |

**Manim animation:** Points from both spirals being "pushed" through learned flow field over time (t=0 to t=1), separating into linearly separable clusters. Optionally show decision boundary evolving over training epochs.

**Features demonstrated:**
- `adjoint()` wrapper for memory-efficient gradients
- Differentiability through ODE solve
- Integration with `nn.Module`
- Memory comparison (O(1) vs O(n_steps))

### Section 5: Wrap-up (2 cells)

| Cell | Type | Content |
|------|------|---------|
| 1 | Markdown | Solver comparison table |
| 2 | Markdown | Links to docs, more examples, GitHub |

**Solver comparison table:**

| Solver | Order | Adaptive | Stiff | Use Case |
|--------|-------|----------|-------|----------|
| `euler` | 1 | No | No | Education, quick prototypes |
| `midpoint` | 2 | No | No | Smooth problems, better than Euler |
| `runge_kutta_4` | 4 | No | No | Workhorse for non-stiff problems |
| `dormand_prince_5` | 5(4) | Yes | No | Production-quality, error control |
| `backward_euler` | 1 | No | Yes | Stiff problems |

## Summary

| Section | Cells | Key Features |
|---------|-------|--------------|
| Introduction | 3 | Imports, setup |
| Double Pendulum | 10 | `dormand_prince_5`, adaptive stepping, interpolant |
| Lotka-Volterra | 10 | TensorDict, solver comparison, `runge_kutta_4` |
| Neural ODE | 10 | `adjoint()`, differentiability, memory efficiency |
| Wrap-up | 2 | Reference table, links |
| **Total** | **~35** | |

## Status

- [x] Design complete
- [ ] Implementation
