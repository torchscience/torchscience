# Optimization Module Roadmap

**Date:** 2025-12-31
**Status:** Design Complete

## Overview

This document outlines the roadmap for `torchscience.optimization`, focusing on practical and research-oriented optimization operators that complement PyTorch's `torch.optim` (which handles neural network training). This module targets "inner loop" optimization: solving equations, fitting subproblems, and constraints within larger differentiable pipelines.

## Design Principles

1. **Differentiability critical** - All operators support autograd where mathematically possible (implicit differentiation for iterative solvers)
2. **Batched execution** - Solve N independent problems in parallel
3. **Complementary to torch.optim** - Focus on scientific computing optimization, not neural network training
4. **Wikipedia-aligned naming** - Operator names match Wikipedia article conventions

## Module Structure

```
torchscience.optimization/
├── root_finding/
├── minimization/
├── constrained/
├── combinatorial/
└── test_functions/
```

## Submodules

### `root_finding/`

Find x where f(x) = 0.

| Operator | Wikipedia Article | Status |
|----------|------------------|--------|
| `brent` | [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method) | Done |
| `newton` | [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) | Planned |
| `fixed_point` | [Fixed-point iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration) | Planned |
| `bisection` | [Bisection method](https://en.wikipedia.org/wiki/Bisection_method) | Planned |

**Differentiability:** Implicit differentiation. If f(x*, theta) = 0, then dx*/dtheta = -[df/dx]^{-1} * df/dtheta.

**Priority:** `newton` (multivariate workhorse)

### `minimization/`

Find x that minimizes f(x).

| Operator | Wikipedia Article | Status |
|----------|------------------|--------|
| `levenberg_marquardt` | [Levenberg-Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) | Planned |
| `nelder_mead` | [Nelder-Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) | Planned |
| `gauss_newton` | [Gauss-Newton algorithm](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) | Planned |

**Differentiability:** Implicit differentiation through optimality conditions.

**Priority:** `levenberg_marquardt` (standard for curve fitting and parameter estimation)

### `constrained/`

Optimization with constraints.

| Operator | Wikipedia Article | Status |
|----------|------------------|--------|
| `augmented_lagrangian` | [Augmented Lagrangian method](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method) | Planned |
| `proximal_gradient` | [Proximal gradient method](https://en.wikipedia.org/wiki/Proximal_gradient_method) | Planned |
| `proximal` | [Proximal operator](https://en.wikipedia.org/wiki/Proximal_operator) | Planned |
| `sequential_quadratic_programming` | [Sequential quadratic programming](https://en.wikipedia.org/wiki/Sequential_quadratic_programming) | Planned |
| `interior_point` | [Interior-point method](https://en.wikipedia.org/wiki/Interior-point_method) | Planned |

**Differentiability:** Implicit differentiation through KKT conditions.

**Priority:** `augmented_lagrangian` (general purpose), `proximal_gradient` (useful for sparsity/regularization)

### `combinatorial/`

Discrete optimization with differentiable gradients.

| Operator | Wikipedia Article | Status |
|----------|------------------|--------|
| `hungarian` | [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) | Planned |
| `sinkhorn` | [Sinkhorn's theorem](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem) | Planned |

**Differentiability:**
- `hungarian`: Implicit differentiation w.r.t. cost matrix
- `sinkhorn`: Naturally differentiable (entropy-regularized optimal transport)

**Priority:** `sinkhorn` (naturally differentiable, widely used)

### `test_functions/`

Benchmark functions for optimization research.

| Function | Wikipedia Article | Status |
|----------|------------------|--------|
| `rosenbrock` | [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) | Done |
| `rastrigin` | [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function) | Planned |
| `ackley` | [Ackley function](https://en.wikipedia.org/wiki/Ackley_function) | Planned |
| `himmelblau` | [Himmelblau's function](https://en.wikipedia.org/wiki/Himmelblau%27s_function) | Planned |
| `shekel` | [Shekel function](https://en.wikipedia.org/wiki/Shekel_function) | Planned |
| `beale` | [Beale function](https://en.wikipedia.org/wiki/Beale_function) | Planned |
| `goldstein_price` | [Goldstein-Price function](https://en.wikipedia.org/wiki/Goldstein%E2%80%93Price_function) | Planned |
| `griewank` | [Griewank function](https://en.wikipedia.org/wiki/Griewank_function) | Planned |

**Priority:** `sphere` (simplest baseline), `rastrigin`, `ackley` (commonly cited)

## Operator Count

- **Existing:** 2 (`brent`, `rosenbrock`)
- **Planned:** 19
- **Total:** 21

## Implementation Notes

### Common Patterns

All optimization operators should follow these patterns:

1. **Batched inputs** - Support solving N independent problems in parallel
2. **Implicit differentiation** - Use `torch.autograd.Function` with custom backward
3. **Dtype-aware tolerances** - Default tolerances based on input dtype precision
4. **Device agnostic** - Work on CPU and CUDA
5. **Vectorized functions** - Accept callable that maps `(N,) -> (N,)` or `(N, d) -> (N,)`

### Implicit Differentiation Template

For iterative solvers that find x* satisfying some optimality condition F(x*, theta) = 0:

```python
class ImplicitDiffFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, result, ...):
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # dx*/dtheta = -[dF/dx]^{-1} * dF/dtheta
        ...
```

## References

- [Constrained optimization](https://en.wikipedia.org/wiki/Constrained_optimization)
- [Test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
- [Combinatorial optimization](https://en.wikipedia.org/wiki/Combinatorial_optimization)
