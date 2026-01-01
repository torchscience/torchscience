# Optimization Module Milestone 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish implementation patterns in the `minimization/`, `combinatorial/`, and `constrained/` submodules.

**Architecture:**
- `levenberg_marquardt`: Pure Python with implicit differentiation (requires function evaluation through autograd)
- `sinkhorn`: C++ implementation (operates on tensors only, no function evaluation)
- `augmented_lagrangian`: Pure Python with implicit differentiation through KKT conditions

**Tech Stack:** PyTorch, torch.func, torch.autograd.Function, ATen, TORCH_LIBRARY

---

## Milestone 1 Operators

| Submodule | Operator | Implementation | Pattern |
|-----------|----------|----------------|---------|
| `minimization/` | `levenberg_marquardt` | Pure Python | Iterative solver with implicit diff |
| `combinatorial/` | `sinkhorn` | C++ | Iterative, naturally differentiable |
| `constrained/` | `augmented_lagrangian` | Pure Python | General constrained solver with KKT implicit diff |

**Already complete:**
- `test_functions/rosenbrock` - reduction pattern established
- `root_finding/brent` - bracketing solver pattern established

---

## Task 1: Levenberg-Marquardt Algorithm

The Levenberg-Marquardt algorithm is a standard method for nonlinear least squares optimization, widely used for curve fitting and parameter estimation.

**Mathematical definition:**
```
Minimize ||r(x)||² where r: R^n → R^m is a residual function

Update step: (J^T J + μI) δ = -J^T r
             x_{k+1} = x_k + δ

Implicit gradient: At optimum J^T r = 0
                   dx*/dθ = -(J^T J)^{-1} J^T (∂r/∂θ)
```

**Files:**
- Create: `src/torchscience/optimization/minimization/__init__.py`
- Create: `src/torchscience/optimization/minimization/_levenberg_marquardt.py`
- Modify: `src/torchscience/optimization/__init__.py`
- Create: `tests/torchscience/optimization/minimization/__init__.py`
- Create: `tests/torchscience/optimization/minimization/test__levenberg_marquardt.py`

### Step 1: Create test directory structure

```bash
mkdir -p tests/torchscience/optimization/minimization
touch tests/torchscience/optimization/minimization/__init__.py
```

### Step 2: Write failing test

Create `tests/torchscience/optimization/minimization/test__levenberg_marquardt.py`:

```python
import pytest
import torch
import torch.testing

import torchscience.optimization.minimization


class TestLevenbergMarquardt:
    def test_linear_least_squares(self):
        """Fit y = ax + b to data."""
        x_data = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y_data = 2.0 * x_data + 1.0  # True: a=2, b=1

        def residuals(params):
            a, b = params[0], params[1]
            return a * x_data + b - y_data

        params0 = torch.tensor([0.0, 0.0])
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, params0
        )
        expected = torch.tensor([2.0, 1.0])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_nonlinear_exponential(self):
        """Fit y = a * exp(-b * x) to data."""
        x_data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        y_data = 2.0 * torch.exp(-0.5 * x_data)

        def residuals(params):
            a, b = params[0], params[1]
            return a * torch.exp(-b * x_data) - y_data

        params0 = torch.tensor([1.0, 1.0])
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, params0
        )
        expected = torch.tensor([2.0, 0.5])
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_implicit_differentiation(self):
        """Test gradient through optimizer via implicit diff."""
        target = torch.tensor([3.0], requires_grad=True)

        def residuals(x):
            return x - target

        x0 = torch.tensor([0.0])
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, x0
        )
        result.sum().backward()
        # dx*/dtarget = 1 (optimal x equals target)
        torch.testing.assert_close(
            target.grad, torch.tensor([1.0]), atol=1e-5, rtol=1e-5
        )

    def test_rosenbrock_minimization(self):
        """Minimize Rosenbrock as least squares: r1 = a-x, r2 = sqrt(b)*(y-x^2)."""
        a, b = 1.0, 100.0

        def residuals(params):
            x, y = params[0], params[1]
            r1 = a - x
            r2 = (b**0.5) * (y - x**2)
            return torch.stack([r1, r2])

        params0 = torch.tensor([-1.0, 1.0])
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, params0, maxiter=200
        )
        expected = torch.tensor([1.0, 1.0])
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_overdetermined_system(self):
        """Test overdetermined system (more residuals than parameters)."""
        # Fit line to 10 points
        x_data = torch.linspace(0, 1, 10)
        y_data = 2.0 * x_data + 1.0 + 0.01 * torch.randn(10)

        def residuals(params):
            a, b = params[0], params[1]
            return a * x_data + b - y_data

        params0 = torch.zeros(2)
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, params0
        )
        # Should be close to [2, 1]
        assert abs(result[0].item() - 2.0) < 0.1
        assert abs(result[1].item() - 1.0) < 0.1

    def test_convergence_failure_warning(self):
        """Test that maxiter=1 doesn't crash (may not converge)."""
        def residuals(x):
            return x - torch.tensor([100.0])

        x0 = torch.tensor([0.0])
        # Should run without error even if not converged
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, x0, maxiter=1
        )
        assert result.shape == x0.shape


class TestLevenbergMarquardtDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""
        def residuals(x):
            return x - torch.tensor([1.0], dtype=dtype)

        x0 = torch.zeros(1, dtype=dtype)
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, x0
        )
        assert result.dtype == dtype
```

### Step 3: Run test to verify it fails

```bash
uv run pytest tests/torchscience/optimization/minimization/test__levenberg_marquardt.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'torchscience.optimization.minimization'`

### Step 4: Create minimization module directory

```bash
mkdir -p src/torchscience/optimization/minimization
```

### Step 5: Implement levenberg_marquardt

Create `src/torchscience/optimization/minimization/_levenberg_marquardt.py`:

```python
from typing import Callable, Optional

import torch
from torch import Tensor


class _LMImplicitGrad(torch.autograd.Function):
    """
    Implicit differentiation through Levenberg-Marquardt optimum.

    At the optimum x*, the gradient of the loss is zero: J^T r = 0
    where J is the Jacobian of residuals and r is the residual vector.

    Using implicit differentiation:
        d(J^T r)/dx * dx/dθ + d(J^T r)/dθ = 0

    For least squares, this simplifies to:
        dx*/dθ = -(J^T J)^{-1} J^T (∂r/∂θ)
    """

    @staticmethod
    def forward(
        ctx,
        result: Tensor,
        residuals_callable: Callable[[Tensor], Tensor],
    ) -> Tensor:
        ctx.residuals_callable = residuals_callable
        ctx.save_for_backward(result)
        return result.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (result,) = ctx.saved_tensors

        with torch.enable_grad():
            x = result.detach().requires_grad_(True)
            r = ctx.residuals_callable(x)

            # Compute Jacobian J = ∂r/∂x
            J = torch.func.jacobian(ctx.residuals_callable)(x)

            # Ensure J is 2D: (num_residuals, num_params)
            if J.dim() == 1:
                J = J.unsqueeze(0)

            # Solve (J^T J) v = J^T grad_output for implicit gradient
            # grad_output is the gradient w.r.t. x*, shape (num_params,)
            JtJ = J.T @ J
            Jt_grad = J.T @ grad_output

            # Add regularization for numerical stability
            reg = 1e-6 * torch.eye(JtJ.shape[0], dtype=JtJ.dtype, device=JtJ.device)
            v = torch.linalg.solve(JtJ + reg, Jt_grad)

            # Backpropagate through residuals with -v as the gradient
            # This computes -v^T (∂r/∂θ) which gives dx*/dθ
            r.backward(-J @ v)

        return None, None


def levenberg_marquardt(
    residuals: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    jacobian: Optional[Callable[[Tensor], Tensor]] = None,
    tol: Optional[float] = None,
    maxiter: int = 100,
    damping: float = 1e-3,
) -> Tensor:
    r"""
    Levenberg-Marquardt algorithm for nonlinear least squares.

    Finds parameters x that minimize the sum of squared residuals:

    .. math::

        \min_x \|r(x)\|^2 = \min_x \sum_i r_i(x)^2

    The algorithm interpolates between Gauss-Newton (fast near optimum)
    and gradient descent (robust far from optimum) using an adaptive
    damping parameter.

    Parameters
    ----------
    residuals : Callable[[Tensor], Tensor]
        Residual function. Takes parameters of shape ``(n,)`` and returns
        residuals of shape ``(m,)`` where ``m >= n``.
    x0 : Tensor
        Initial parameter guess of shape ``(n,)``.
    jacobian : Callable, optional
        Jacobian of residuals. If None, computed via ``torch.func.jacobian``.
        Should return a tensor of shape ``(m, n)``.
    tol : float, optional
        Convergence tolerance on gradient norm. Default: ``sqrt(eps)`` for dtype.
    maxiter : int
        Maximum number of iterations. Default: 100.
    damping : float
        Initial Levenberg-Marquardt damping parameter. Default: 1e-3.

    Returns
    -------
    Tensor
        Optimized parameters of shape ``(n,)``.

    Examples
    --------
    Fit a line y = ax + b to data:

    >>> x_data = torch.tensor([0., 1., 2., 3.])
    >>> y_data = torch.tensor([1., 3., 5., 7.])  # y = 2x + 1
    >>> def residuals(params):
    ...     a, b = params[0], params[1]
    ...     return a * x_data + b - y_data
    >>> result = levenberg_marquardt(residuals, torch.zeros(2))
    >>> result
    tensor([2., 1.])

    The optimizer supports implicit differentiation:

    >>> target = torch.tensor([5.0], requires_grad=True)
    >>> def residuals(x):
    ...     return x - target
    >>> result = levenberg_marquardt(residuals, torch.zeros(1))
    >>> result.backward()
    >>> target.grad
    tensor([1.])

    References
    ----------
    - Levenberg, K. "A method for the solution of certain non-linear
      problems in least squares." Quarterly of applied mathematics 2.2
      (1944): 164-168.
    - Marquardt, D.W. "An algorithm for least-squares estimation of
      nonlinear parameters." Journal of the society for Industrial and
      Applied Mathematics 11.2 (1963): 431-441.

    See Also
    --------
    https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    x = x0.clone()
    mu = damping
    n = x.numel()

    for _ in range(maxiter):
        r = residuals(x)

        if jacobian is not None:
            J = jacobian(x)
        else:
            J = torch.func.jacobian(residuals)(x)

        # Ensure J is 2D
        if J.dim() == 1:
            J = J.unsqueeze(0)

        # Gradient: g = J^T @ r
        g = J.T @ r

        # Check convergence
        if torch.norm(g) < tol:
            break

        # Hessian approximation: H = J^T @ J + mu * I
        JtJ = J.T @ J
        H = JtJ + mu * torch.eye(n, dtype=x.dtype, device=x.device)

        # Solve H @ delta = -g
        try:
            delta = torch.linalg.solve(H, -g)
        except RuntimeError:
            # Matrix is singular, increase damping
            mu *= 10
            continue

        # Compute actual vs predicted reduction
        x_new = x + delta
        r_new = residuals(x_new)

        actual_reduction = torch.sum(r**2) - torch.sum(r_new**2)
        predicted_reduction = -2 * (g @ delta) - delta @ JtJ @ delta

        # Avoid division by zero
        rho = actual_reduction / (predicted_reduction + 1e-10)

        if rho > 0.25:
            # Good step, accept and decrease damping
            x = x_new
            mu = max(mu / 3, 1e-10)
        else:
            # Bad step, reject and increase damping
            mu = min(mu * 2, 1e10)

    # Attach implicit gradient for backpropagation
    return _LMImplicitGrad.apply(x, residuals)
```

### Step 6: Create module __init__.py

Create `src/torchscience/optimization/minimization/__init__.py`:

```python
from ._levenberg_marquardt import levenberg_marquardt

__all__ = [
    "levenberg_marquardt",
]
```

### Step 7: Update optimization __init__.py

Modify `src/torchscience/optimization/__init__.py`:

```python
from . import minimization, root_finding, test_functions

__all__ = [
    "minimization",
    "root_finding",
    "test_functions",
]
```

### Step 8: Run tests

```bash
uv run pytest tests/torchscience/optimization/minimization/test__levenberg_marquardt.py -v
```

Expected: All tests pass

### Step 9: Commit

```bash
git add src/torchscience/optimization/minimization/ \
        src/torchscience/optimization/__init__.py \
        tests/torchscience/optimization/minimization/
git commit -m "feat(minimization): add levenberg_marquardt algorithm

Implement Levenberg-Marquardt for nonlinear least squares:
- Adaptive damping between Gauss-Newton and gradient descent
- Implicit differentiation through optimum for autograd
- Pure Python implementation using torch.func.jacobian"
```

---

## Task 2: Sinkhorn Algorithm

The Sinkhorn algorithm computes entropy-regularized optimal transport plans. Unlike Levenberg-Marquardt, Sinkhorn operates on tensors only (no function evaluation), so it's implemented in C++.

**Mathematical definition:**
```
Minimize <C, P> + ε H(P)
subject to: P1 = a, P^T 1 = b, P >= 0

Solution via scaling iterations:
    K = exp(-C/ε)
    u^{k+1} = a / (K v^k)
    v^{k+1} = b / (K^T u^{k+1})
    P = diag(u) K diag(v)

Gradient: dL/dC = -P/ε (element-wise)
```

**Files:**
- Create: `src/torchscience/optimization/combinatorial/__init__.py`
- Create: `src/torchscience/optimization/combinatorial/_sinkhorn.py`
- Create: `src/torchscience/csrc/cpu/optimization/combinatorial.h`
- Create: `src/torchscience/csrc/meta/optimization/combinatorial.h`
- Create: `src/torchscience/csrc/autograd/optimization/combinatorial.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Modify: `src/torchscience/optimization/__init__.py`
- Create: `tests/torchscience/optimization/combinatorial/__init__.py`
- Create: `tests/torchscience/optimization/combinatorial/test__sinkhorn.py`

### Step 1: Create test directory structure

```bash
mkdir -p tests/torchscience/optimization/combinatorial
touch tests/torchscience/optimization/combinatorial/__init__.py
```

### Step 2: Write failing test

Create `tests/torchscience/optimization/combinatorial/test__sinkhorn.py`:

```python
import pytest
import torch
import torch.testing

import torchscience.optimization.combinatorial


class TestSinkhorn:
    def test_uniform_marginals(self):
        """Test with uniform marginals."""
        n, m = 3, 4
        C = torch.rand(n, m)
        a = torch.ones(n) / n
        b = torch.ones(m) / m

        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)

        # Check marginal constraints
        torch.testing.assert_close(P.sum(dim=-1), a, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(P.sum(dim=-2), b, atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        """Test output shape matches cost matrix."""
        C = torch.rand(5, 7)
        a = torch.ones(5) / 5
        b = torch.ones(7) / 7
        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        assert P.shape == C.shape

    def test_batched(self):
        """Test batched cost matrices."""
        batch, n, m = 2, 3, 4
        C = torch.rand(batch, n, m)
        a = torch.ones(batch, n) / n
        b = torch.ones(batch, m) / m

        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)

        assert P.shape == (batch, n, m)
        torch.testing.assert_close(P.sum(dim=-1), a, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(P.sum(dim=-2), b, atol=1e-4, rtol=1e-4)

    def test_nonnegative(self):
        """Test that transport plan is non-negative."""
        C = torch.rand(5, 5)
        a = torch.ones(5) / 5
        b = torch.ones(5) / 5
        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        assert (P >= -1e-6).all()

    def test_gradient_wrt_cost(self):
        """Test gradient with respect to cost matrix."""
        n, m = 3, 4
        C = torch.rand(n, m, requires_grad=True)
        a = torch.ones(n) / n
        b = torch.ones(m) / m

        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        loss = (P * C).sum()
        loss.backward()

        assert C.grad is not None
        assert C.grad.shape == C.shape

    def test_gradcheck(self):
        """Test gradient correctness via finite differences."""
        n, m = 3, 4
        C = torch.rand(n, m, dtype=torch.float64, requires_grad=True)
        a = torch.ones(n, dtype=torch.float64) / n
        b = torch.ones(m, dtype=torch.float64) / m

        def fn(C_):
            return torchscience.optimization.combinatorial.sinkhorn(C_, a, b)

        assert torch.autograd.gradcheck(fn, (C,), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_regularization_effect(self):
        """Test that smaller epsilon gives sparser solution."""
        n = 3
        C = torch.tensor(
            [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
        )
        a = torch.ones(n) / n
        b = torch.ones(n) / n

        P_large_eps = torchscience.optimization.combinatorial.sinkhorn(
            C, a, b, epsilon=1.0
        )
        P_small_eps = torchscience.optimization.combinatorial.sinkhorn(
            C, a, b, epsilon=0.01
        )

        # Smaller epsilon should be more peaked (higher max value)
        assert P_small_eps.max() > P_large_eps.max()

    def test_identity_cost(self):
        """With zero cost, uniform marginals give uniform plan."""
        n = 4
        C = torch.zeros(n, n)
        a = torch.ones(n) / n
        b = torch.ones(n) / n
        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        expected = torch.ones(n, n) / (n * n)
        torch.testing.assert_close(P, expected, atol=1e-4, rtol=1e-4)

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        C = torch.empty(3, 4, device="meta")
        a = torch.empty(3, device="meta")
        b = torch.empty(4, device="meta")
        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        assert P.shape == (3, 4)
        assert P.device.type == "meta"


class TestSinkhornDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""
        C = torch.rand(3, 4, dtype=dtype)
        a = torch.ones(3, dtype=dtype) / 3
        b = torch.ones(4, dtype=dtype) / 4
        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        assert P.dtype == dtype
```

### Step 3: Run test to verify it fails

```bash
uv run pytest tests/torchscience/optimization/combinatorial/test__sinkhorn.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'torchscience.optimization.combinatorial'`

### Step 4: Add schema registration

Modify `src/torchscience/csrc/torchscience.cpp`, add in TORCH_LIBRARY block:

```cpp
  // optimization.combinatorial
  module.def("sinkhorn(Tensor C, Tensor a, Tensor b, float epsilon, int maxiter, float tol) -> Tensor");
  module.def("sinkhorn_backward(Tensor grad_output, Tensor P, Tensor C, float epsilon) -> Tensor");
```

### Step 5: Create CPU kernel

Create `src/torchscience/csrc/cpu/optimization/combinatorial.h`:

```cpp
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::cpu::optimization::combinatorial {

inline at::Tensor sinkhorn(
    const at::Tensor& C,
    const at::Tensor& a,
    const at::Tensor& b,
    double epsilon,
    int64_t maxiter,
    double tol
) {
    TORCH_CHECK(C.dim() >= 2, "Cost matrix must have at least 2 dimensions");
    TORCH_CHECK(a.dim() >= 1, "Source marginal must have at least 1 dimension");
    TORCH_CHECK(b.dim() >= 1, "Target marginal must have at least 1 dimension");
    TORCH_CHECK(epsilon > 0, "Regularization epsilon must be positive");

    // Compute kernel matrix K = exp(-C / epsilon)
    at::Tensor K = at::exp(-C / epsilon);

    // Initialize scaling vectors
    at::Tensor u = at::ones_like(a);
    at::Tensor v = at::ones_like(b);

    for (int64_t iter = 0; iter < maxiter; ++iter) {
        at::Tensor u_prev = u.clone();

        // v = b / (K^T @ u)
        // K^T @ u: (..., m, n) @ (..., n) -> (..., m)
        at::Tensor Ktu = at::matmul(
            K.transpose(-2, -1),
            u.unsqueeze(-1)
        ).squeeze(-1);
        v = b / at::clamp_min(Ktu, 1e-10);

        // u = a / (K @ v)
        // K @ v: (..., n, m) @ (..., m) -> (..., n)
        at::Tensor Kv = at::matmul(K, v.unsqueeze(-1)).squeeze(-1);
        u = a / at::clamp_min(Kv, 1e-10);

        // Check convergence
        double max_diff = at::max(at::abs(u - u_prev)).item<double>();
        if (max_diff < tol) {
            break;
        }
    }

    // Transport plan: P = diag(u) @ K @ diag(v)
    // Equivalent to: P_ij = u_i * K_ij * v_j
    at::Tensor P = u.unsqueeze(-1) * K * v.unsqueeze(-2);

    return P;
}

inline at::Tensor sinkhorn_backward(
    const at::Tensor& grad_output,
    const at::Tensor& P,
    const at::Tensor& C,
    double epsilon
) {
    // Gradient w.r.t. C:
    // P = diag(u) @ K @ diag(v), where K = exp(-C/epsilon)
    // dP/dC = -P / epsilon (element-wise, since dK/dC = -K/epsilon)
    // dL/dC = dL/dP * dP/dC = grad_output * (-P / epsilon)
    return -grad_output * P / epsilon;
}

}  // namespace torchscience::cpu::optimization::combinatorial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "sinkhorn",
        &torchscience::cpu::optimization::combinatorial::sinkhorn
    );
    module.impl(
        "sinkhorn_backward",
        &torchscience::cpu::optimization::combinatorial::sinkhorn_backward
    );
}
```

### Step 6: Create Meta kernel

Create `src/torchscience/csrc/meta/optimization/combinatorial.h`:

```cpp
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::optimization::combinatorial {

inline at::Tensor sinkhorn(
    const at::Tensor& C,
    const at::Tensor& a,
    const at::Tensor& b,
    double epsilon,
    int64_t maxiter,
    double tol
) {
    TORCH_CHECK(C.dim() >= 2, "Cost matrix must have at least 2 dimensions");

    // Output shape is same as C
    return at::empty(C.sizes(), C.options().device(at::kMeta));
}

inline at::Tensor sinkhorn_backward(
    const at::Tensor& grad_output,
    const at::Tensor& P,
    const at::Tensor& C,
    double epsilon
) {
    return at::empty(C.sizes(), C.options().device(at::kMeta));
}

}  // namespace torchscience::meta::optimization::combinatorial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "sinkhorn",
        &torchscience::meta::optimization::combinatorial::sinkhorn
    );
    module.impl(
        "sinkhorn_backward",
        &torchscience::meta::optimization::combinatorial::sinkhorn_backward
    );
}
```

### Step 7: Create Autograd wrapper

Create `src/torchscience/csrc/autograd/optimization/combinatorial.h`:

```cpp
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::optimization::combinatorial {

class Sinkhorn : public torch::autograd::Function<Sinkhorn> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& C,
        const at::Tensor& a,
        const at::Tensor& b,
        double epsilon,
        int64_t maxiter,
        double tol
    ) {
        at::AutoDispatchBelowAutograd guard;

        at::Tensor P = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::sinkhorn", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                double,
                int64_t,
                double
            )>()
            .call(C, a, b, epsilon, maxiter, tol);

        context->save_for_backward({P, C});
        context->saved_data["epsilon"] = epsilon;

        return P;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& gradient_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        double epsilon = context->saved_data["epsilon"].toDouble();

        at::Tensor P = saved[0];
        at::Tensor C = saved[1];
        at::Tensor grad_output = gradient_outputs[0];

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_C = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::sinkhorn_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                double
            )>()
            .call(grad_output, P, C, epsilon);

        // Return gradients for: C, a, b, epsilon, maxiter, tol
        // Only C gets a gradient; others are not differentiable
        return {grad_C, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

inline at::Tensor sinkhorn(
    const at::Tensor& C,
    const at::Tensor& a,
    const at::Tensor& b,
    double epsilon,
    int64_t maxiter,
    double tol
) {
    return Sinkhorn::apply(C, a, b, epsilon, maxiter, tol);
}

}  // namespace torchscience::autograd::optimization::combinatorial

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "sinkhorn",
        &torchscience::autograd::optimization::combinatorial::sinkhorn
    );
}
```

### Step 8: Include headers in torchscience.cpp

Add these includes at the top of `src/torchscience/csrc/torchscience.cpp` with the other includes:

```cpp
#include "autograd/optimization/combinatorial.h"
#include "cpu/optimization/combinatorial.h"
#include "meta/optimization/combinatorial.h"
```

### Step 9: Create Python API

Create `src/torchscience/optimization/combinatorial/_sinkhorn.py`:

```python
import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def sinkhorn(
    C: Tensor,
    a: Tensor,
    b: Tensor,
    *,
    epsilon: float = 0.1,
    maxiter: int = 100,
    tol: float = 1e-6,
) -> Tensor:
    r"""
    Sinkhorn algorithm for entropy-regularized optimal transport.

    Computes the optimal transport plan between source distribution ``a``
    and target distribution ``b`` with cost matrix ``C``, regularized by
    entropy to encourage smooth solutions.

    Solves the optimization problem:

    .. math::

        \min_P \langle C, P \rangle + \epsilon H(P)
        \quad \text{s.t.} \quad P \mathbf{1} = a, \; P^T \mathbf{1} = b, \; P \geq 0

    where :math:`H(P) = -\sum_{ij} P_{ij} \log P_{ij}` is the entropy.

    Parameters
    ----------
    C : Tensor
        Cost matrix of shape ``(..., n, m)``. Entry ``C[i,j]`` is the cost
        of transporting mass from source ``i`` to target ``j``.
    a : Tensor
        Source marginal of shape ``(..., n)``. Must be non-negative and
        sum to 1 (probability distribution).
    b : Tensor
        Target marginal of shape ``(..., m)``. Must be non-negative and
        sum to 1 (probability distribution).
    epsilon : float, optional
        Entropy regularization strength. Smaller values give solutions
        closer to unregularized optimal transport but may converge slower.
        Default: 0.1.
    maxiter : int, optional
        Maximum number of Sinkhorn iterations. Default: 100.
    tol : float, optional
        Convergence tolerance on scaling vector change. Default: 1e-6.

    Returns
    -------
    Tensor
        Transport plan ``P`` of shape ``(..., n, m)``. Entry ``P[i,j]``
        is the amount of mass transported from source ``i`` to target ``j``.
        Satisfies ``P.sum(dim=-1) == a`` and ``P.sum(dim=-2) == b``.

    Examples
    --------
    Compute transport plan between uniform distributions:

    >>> C = torch.rand(3, 4)
    >>> a = torch.ones(3) / 3
    >>> b = torch.ones(4) / 4
    >>> P = sinkhorn(C, a, b)
    >>> P.sum(dim=-1)  # Should equal a
    tensor([0.3333, 0.3333, 0.3333])

    The transport cost (objective value) is:

    >>> (C * P).sum()
    tensor(...)

    Gradients flow through the transport plan:

    >>> C = torch.rand(3, 3, requires_grad=True)
    >>> a = torch.ones(3) / 3
    >>> b = torch.ones(3) / 3
    >>> P = sinkhorn(C, a, b)
    >>> (C * P).sum().backward()
    >>> C.grad.shape
    torch.Size([3, 3])

    Notes
    -----
    - The algorithm converges faster with larger ``epsilon`` but gives
      smoother (less sparse) transport plans.
    - For unregularized optimal transport (assignment problem), use
      ``epsilon < 0.01`` but increase ``maxiter``.
    - Gradients are only computed w.r.t. the cost matrix ``C``. The
      marginals ``a`` and ``b`` are treated as constants.

    References
    ----------
    - Cuturi, M. "Sinkhorn distances: Lightspeed computation of optimal
      transport." NeurIPS 2013.
    - Peyré, G. and Cuturi, M. "Computational optimal transport."
      Foundations and Trends in Machine Learning 11.5-6 (2019): 355-607.

    See Also
    --------
    https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem
    """
    return torch.ops.torchscience.sinkhorn(C, a, b, epsilon, maxiter, tol)
```

### Step 10: Create module __init__.py

Create `src/torchscience/optimization/combinatorial/__init__.py`:

```python
from ._sinkhorn import sinkhorn

__all__ = [
    "sinkhorn",
]
```

### Step 11: Update optimization __init__.py

Modify `src/torchscience/optimization/__init__.py`:

```python
from . import combinatorial, minimization, root_finding, test_functions

__all__ = [
    "combinatorial",
    "minimization",
    "root_finding",
    "test_functions",
]
```

### Step 12: Create combinatorial test __init__.py

```bash
mkdir -p tests/torchscience/optimization/combinatorial
touch tests/torchscience/optimization/combinatorial/__init__.py
```

### Step 13: Run tests

```bash
uv run pytest tests/torchscience/optimization/combinatorial/test__sinkhorn.py -v
```

Expected: All tests pass

### Step 14: Commit

```bash
git add src/torchscience/optimization/combinatorial/ \
        src/torchscience/optimization/__init__.py \
        src/torchscience/csrc/cpu/optimization/combinatorial.h \
        src/torchscience/csrc/meta/optimization/combinatorial.h \
        src/torchscience/csrc/autograd/optimization/combinatorial.h \
        src/torchscience/csrc/torchscience.cpp \
        tests/torchscience/optimization/combinatorial/
git commit -m "feat(combinatorial): add sinkhorn optimal transport algorithm

Implement entropy-regularized optimal transport via Sinkhorn iterations:
- CPU kernel with iterative scaling
- Meta kernel for shape inference
- Autograd wrapper with gradient w.r.t. cost matrix
- Batched input support"
```

---

## Task 3: Augmented Lagrangian Method

The augmented Lagrangian method solves constrained optimization problems by converting them to a sequence of unconstrained problems. It handles both equality and inequality constraints.

**Mathematical definition:**
```
Minimize f(x)
subject to: h(x) = 0  (equality constraints)
            g(x) <= 0 (inequality constraints)

Augmented Lagrangian:
    L_ρ(x, λ, μ) = f(x) + λᵀh(x) + (ρ/2)||h(x)||²
                  + Σᵢ (1/2ρ)[max(0, μᵢ + ρgᵢ(x))² - μᵢ²]

Update:
    x_{k+1} = argmin_x L_ρ(x, λ_k, μ_k)  [inner loop: unconstrained minimization]
    λ_{k+1} = λ_k + ρ h(x_{k+1})
    μ_{k+1} = max(0, μ_k + ρ g(x_{k+1}))

Implicit gradient via KKT conditions at optimum:
    ∇f + Jₕᵀλ* + Jᵧᵀμ* = 0
    h(x*) = 0
    μ* ⊙ g(x*) = 0, μ* >= 0, g(x*) <= 0
```

**Files:**
- Create: `src/torchscience/optimization/constrained/__init__.py`
- Create: `src/torchscience/optimization/constrained/_augmented_lagrangian.py`
- Modify: `src/torchscience/optimization/__init__.py`
- Create: `tests/torchscience/optimization/constrained/__init__.py`
- Create: `tests/torchscience/optimization/constrained/test__augmented_lagrangian.py`

### Step 1: Create test directory structure

```bash
mkdir -p tests/torchscience/optimization/constrained
touch tests/torchscience/optimization/constrained/__init__.py
```

### Step 2: Write failing test

Create `tests/torchscience/optimization/constrained/test__augmented_lagrangian.py`:

```python
import pytest
import torch
import torch.testing

import torchscience.optimization.constrained


class TestAugmentedLagrangian:
    def test_unconstrained_quadratic(self):
        """Without constraints, should minimize f(x) = ||x - target||²."""
        target = torch.tensor([1.0, 2.0])

        def objective(x):
            return torch.sum((x - target) ** 2)

        x0 = torch.zeros(2)
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective, x0
        )
        torch.testing.assert_close(result, target, atol=1e-4, rtol=1e-4)

    def test_equality_constraint(self):
        """Minimize x² + y² subject to x + y = 1."""

        def objective(x):
            return torch.sum(x**2)

        def eq_constraints(x):
            return x.sum() - 1.0  # h(x) = x + y - 1 = 0

        x0 = torch.zeros(2)
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective, x0, eq_constraints=eq_constraints
        )
        # Optimal: x = y = 0.5
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_inequality_constraint(self):
        """Minimize -x subject to x <= 2."""

        def objective(x):
            return -x.sum()

        def ineq_constraints(x):
            return x - 2.0  # g(x) = x - 2 <= 0

        x0 = torch.zeros(1)
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective, x0, ineq_constraints=ineq_constraints
        )
        # Optimal: x = 2 (at the constraint boundary)
        expected = torch.tensor([2.0])
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_mixed_constraints(self):
        """Minimize x² + y² subject to x + y = 1, x >= 0.3."""

        def objective(x):
            return torch.sum(x**2)

        def eq_constraints(x):
            return x.sum() - 1.0

        def ineq_constraints(x):
            return 0.3 - x[0]  # -x + 0.3 <= 0, i.e., x >= 0.3

        x0 = torch.tensor([0.5, 0.5])
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective,
            x0,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        # Without inequality: x=y=0.5. With x>=0.3, constraint is inactive.
        # If we had x>=0.6, then x=0.6, y=0.4
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_active_inequality(self):
        """Minimize x² + y² subject to x + y = 1, x >= 0.6."""

        def objective(x):
            return torch.sum(x**2)

        def eq_constraints(x):
            return x.sum() - 1.0

        def ineq_constraints(x):
            return 0.6 - x[0]  # x >= 0.6

        x0 = torch.tensor([0.7, 0.3])
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective,
            x0,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        # Constrained optimum: x=0.6, y=0.4
        expected = torch.tensor([0.6, 0.4])
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_implicit_differentiation(self):
        """Test gradient through constrained optimizer."""
        target = torch.tensor([1.0], requires_grad=True)

        def objective(x):
            return (x - target) ** 2

        def eq_constraints(x):
            return x - 0.5  # x = 0.5

        x0 = torch.zeros(1)
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective, x0, eq_constraints=eq_constraints
        )
        # Result is always 0.5 regardless of target, so gradient should be 0
        loss = result.sum()
        loss.backward()
        torch.testing.assert_close(
            target.grad, torch.tensor([0.0]), atol=1e-4, rtol=1e-4
        )

    def test_rosenbrock_constrained(self):
        """Minimize Rosenbrock subject to x² + y² <= 2."""

        def objective(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        def ineq_constraints(x):
            return torch.sum(x**2) - 2.0  # ||x||² <= 2

        x0 = torch.tensor([0.0, 0.0])
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective, x0, ineq_constraints=ineq_constraints, maxiter=50
        )
        # Unconstrained optimum is (1, 1) with ||x||² = 2, exactly on boundary
        expected = torch.tensor([1.0, 1.0])
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


class TestAugmentedLagrangianDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def objective(x):
            return torch.sum(x**2)

        x0 = torch.ones(2, dtype=dtype)
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective, x0
        )
        assert result.dtype == dtype
```

### Step 3: Run test to verify it fails

```bash
uv run pytest tests/torchscience/optimization/constrained/test__augmented_lagrangian.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'torchscience.optimization.constrained'`

### Step 4: Create constrained module directory

```bash
mkdir -p src/torchscience/optimization/constrained
```

### Step 5: Implement augmented_lagrangian

Create `src/torchscience/optimization/constrained/_augmented_lagrangian.py`:

```python
from typing import Callable, Optional

import torch
from torch import Tensor


class _ALImplicitGrad(torch.autograd.Function):
    """
    Implicit differentiation through augmented Lagrangian optimum via KKT conditions.

    At the optimum (x*, λ*, μ*), the KKT conditions hold:
        ∇f(x*) + Jₕᵀλ* + Jᵧᵀμ* = 0  (stationarity)
        h(x*) = 0                      (primal feasibility - equality)
        g(x*) <= 0, μ* >= 0            (primal feasibility - inequality)
        μ* ⊙ g(x*) = 0                 (complementary slackness)

    Differentiating these conditions gives the implicit gradient.
    """

    @staticmethod
    def forward(
        ctx,
        result: Tensor,
        objective: Callable[[Tensor], Tensor],
        eq_constraints: Optional[Callable[[Tensor], Tensor]],
        ineq_constraints: Optional[Callable[[Tensor], Tensor]],
        lambda_eq: Optional[Tensor],
        mu_ineq: Optional[Tensor],
    ) -> Tensor:
        ctx.objective = objective
        ctx.eq_constraints = eq_constraints
        ctx.ineq_constraints = ineq_constraints
        ctx.save_for_backward(result, lambda_eq, mu_ineq)
        return result.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        result, lambda_eq, mu_ineq = ctx.saved_tensors

        with torch.enable_grad():
            x = result.detach().requires_grad_(True)

            # Compute gradient of Lagrangian w.r.t. x
            f = ctx.objective(x)
            grad_f = torch.autograd.grad(f, x, create_graph=True)[0]

            total_grad = grad_f.clone()

            if ctx.eq_constraints is not None and lambda_eq is not None:
                h = ctx.eq_constraints(x)
                if h.dim() == 0:
                    h = h.unsqueeze(0)
                for i in range(h.numel()):
                    grad_hi = torch.autograd.grad(
                        h.flatten()[i], x, retain_graph=True, create_graph=True
                    )[0]
                    total_grad = total_grad + lambda_eq.flatten()[i] * grad_hi

            if ctx.ineq_constraints is not None and mu_ineq is not None:
                g = ctx.ineq_constraints(x)
                if g.dim() == 0:
                    g = g.unsqueeze(0)
                for i in range(g.numel()):
                    if mu_ineq.flatten()[i] > 1e-8:  # Active constraint
                        grad_gi = torch.autograd.grad(
                            g.flatten()[i], x, retain_graph=True, create_graph=True
                        )[0]
                        total_grad = total_grad + mu_ineq.flatten()[i] * grad_gi

            # Compute Hessian of Lagrangian (for implicit function theorem)
            hess_rows = []
            for i in range(x.numel()):
                grad_i = torch.autograd.grad(
                    total_grad.flatten()[i],
                    x,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if grad_i is None:
                    grad_i = torch.zeros_like(x)
                hess_rows.append(grad_i.flatten())

            H = torch.stack(hess_rows)

            # Add regularization for numerical stability
            reg = 1e-4 * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
            H = H + reg

            # Solve H @ v = grad_output for implicit gradient
            try:
                v = torch.linalg.solve(H, grad_output.flatten())
            except RuntimeError:
                v = torch.linalg.lstsq(H, grad_output.flatten()).solution

            # Backpropagate through objective
            f.backward(-v @ grad_f)

        return None, None, None, None, None, None


def augmented_lagrangian(
    objective: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    eq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    ineq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    tol: Optional[float] = None,
    maxiter: int = 50,
    inner_maxiter: int = 100,
    rho: float = 1.0,
    rho_max: float = 1e6,
) -> Tensor:
    r"""
    Augmented Lagrangian method for constrained optimization.

    Solves the constrained optimization problem:

    .. math::

        \min_x f(x) \quad \text{s.t.} \quad h(x) = 0, \; g(x) \leq 0

    by iteratively solving unconstrained subproblems with the augmented
    Lagrangian:

    .. math::

        L_\rho(x, \lambda, \mu) = f(x) + \lambda^T h(x) + \frac{\rho}{2}\|h(x)\|^2
        + \sum_i \frac{1}{2\rho}[\max(0, \mu_i + \rho g_i(x))^2 - \mu_i^2]

    Parameters
    ----------
    objective : Callable[[Tensor], Tensor]
        Objective function f(x) to minimize. Takes parameters of shape ``(n,)``
        and returns a scalar.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    eq_constraints : Callable, optional
        Equality constraint function h(x). Returns tensor where h(x) = 0 is required.
    ineq_constraints : Callable, optional
        Inequality constraint function g(x). Returns tensor where g(x) <= 0 is required.
    tol : float, optional
        Convergence tolerance on constraint violation. Default: ``sqrt(eps)`` for dtype.
    maxiter : int
        Maximum outer iterations (Lagrange multiplier updates). Default: 50.
    inner_maxiter : int
        Maximum inner iterations per subproblem. Default: 100.
    rho : float
        Initial penalty parameter. Default: 1.0.
    rho_max : float
        Maximum penalty parameter. Default: 1e6.

    Returns
    -------
    Tensor
        Optimized parameters of shape ``(n,)``.

    Examples
    --------
    Minimize x² + y² subject to x + y = 1:

    >>> def objective(x):
    ...     return torch.sum(x**2)
    >>> def eq_constraint(x):
    ...     return x.sum() - 1.0
    >>> x0 = torch.zeros(2)
    >>> result = augmented_lagrangian(objective, x0, eq_constraints=eq_constraint)
    >>> result
    tensor([0.5, 0.5])

    With inequality constraints (x >= 0.6):

    >>> def ineq_constraint(x):
    ...     return 0.6 - x[0]  # -x + 0.6 <= 0
    >>> result = augmented_lagrangian(
    ...     objective, x0, eq_constraints=eq_constraint, ineq_constraints=ineq_constraint
    ... )
    >>> result
    tensor([0.6, 0.4])

    References
    ----------
    - Nocedal, J. and Wright, S.J. "Numerical Optimization." Chapter 17.
    - Bertsekas, D.P. "Constrained Optimization and Lagrange Multiplier Methods."

    See Also
    --------
    https://en.wikipedia.org/wiki/Augmented_Lagrangian_method
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    x = x0.clone()
    n = x.numel()

    # Initialize Lagrange multipliers
    lambda_eq = None
    mu_ineq = None

    if eq_constraints is not None:
        h0 = eq_constraints(x0)
        if h0.dim() == 0:
            h0 = h0.unsqueeze(0)
        lambda_eq = torch.zeros(h0.numel(), dtype=x0.dtype, device=x0.device)

    if ineq_constraints is not None:
        g0 = ineq_constraints(x0)
        if g0.dim() == 0:
            g0 = g0.unsqueeze(0)
        mu_ineq = torch.zeros(g0.numel(), dtype=x0.dtype, device=x0.device)

    for outer_iter in range(maxiter):
        # Define augmented Lagrangian for inner minimization
        def augmented_lagrangian_func(x_inner):
            L = objective(x_inner)

            if eq_constraints is not None:
                h = eq_constraints(x_inner)
                if h.dim() == 0:
                    h = h.unsqueeze(0)
                L = L + torch.dot(lambda_eq, h) + (rho / 2) * torch.sum(h**2)

            if ineq_constraints is not None:
                g = ineq_constraints(x_inner)
                if g.dim() == 0:
                    g = g.unsqueeze(0)
                # Powell-Hestenes-Rockafellar formulation
                for i in range(g.numel()):
                    slack = torch.clamp(mu_ineq[i] + rho * g[i], min=0.0)
                    L = L + (1 / (2 * rho)) * (slack**2 - mu_ineq[i] ** 2)

            return L

        # Inner loop: minimize augmented Lagrangian using gradient descent
        x_inner = x.clone().requires_grad_(True)

        for inner_iter in range(inner_maxiter):
            L = augmented_lagrangian_func(x_inner)
            grad = torch.autograd.grad(L, x_inner, create_graph=False)[0]

            if torch.norm(grad) < tol:
                break

            # Simple gradient descent with line search
            alpha = 1.0
            x_new = x_inner.detach() - alpha * grad

            # Backtracking line search
            for _ in range(20):
                with torch.no_grad():
                    L_new = augmented_lagrangian_func(x_new)
                    L_old = augmented_lagrangian_func(x_inner.detach())
                if L_new < L_old:
                    break
                alpha *= 0.5
                x_new = x_inner.detach() - alpha * grad

            x_inner = x_new.requires_grad_(True)

        x = x_inner.detach()

        # Check constraint satisfaction
        max_violation = 0.0

        if eq_constraints is not None:
            h = eq_constraints(x)
            if h.dim() == 0:
                h = h.unsqueeze(0)
            max_violation = max(max_violation, torch.max(torch.abs(h)).item())
            # Update equality multipliers
            lambda_eq = lambda_eq + rho * h.detach()

        if ineq_constraints is not None:
            g = ineq_constraints(x)
            if g.dim() == 0:
                g = g.unsqueeze(0)
            max_violation = max(max_violation, torch.max(torch.clamp(g, min=0)).item())
            # Update inequality multipliers
            mu_ineq = torch.clamp(mu_ineq + rho * g.detach(), min=0.0)

        if max_violation < tol:
            break

        # Increase penalty if constraints not satisfied
        if max_violation > 0.25 * tol:
            rho = min(rho * 2, rho_max)

    # Attach implicit gradient for backpropagation
    return _ALImplicitGrad.apply(
        x, objective, eq_constraints, ineq_constraints, lambda_eq, mu_ineq
    )
```

### Step 6: Create module __init__.py

Create `src/torchscience/optimization/constrained/__init__.py`:

```python
from ._augmented_lagrangian import augmented_lagrangian

__all__ = [
    "augmented_lagrangian",
]
```

### Step 7: Update optimization __init__.py

Modify `src/torchscience/optimization/__init__.py`:

```python
from . import combinatorial, constrained, minimization, root_finding, test_functions

__all__ = [
    "combinatorial",
    "constrained",
    "minimization",
    "root_finding",
    "test_functions",
]
```

### Step 8: Run tests

```bash
uv run pytest tests/torchscience/optimization/constrained/test__augmented_lagrangian.py -v
```

Expected: All tests pass

### Step 9: Commit

```bash
git add src/torchscience/optimization/constrained/ \
        src/torchscience/optimization/__init__.py \
        tests/torchscience/optimization/constrained/
git commit -m "feat(constrained): add augmented_lagrangian method

Implement augmented Lagrangian for constrained optimization:
- Equality constraints (h(x) = 0)
- Inequality constraints (g(x) <= 0)
- Powell-Hestenes-Rockafellar penalty formulation
- Implicit differentiation through KKT conditions
- Adaptive penalty parameter updates"
```

---

## Summary

After completing all three tasks, the optimization module structure is:

```
torchscience.optimization/
├── __init__.py
├── combinatorial/
│   ├── __init__.py
│   └── _sinkhorn.py              [NEW]
├── constrained/
│   ├── __init__.py               [NEW]
│   └── _augmented_lagrangian.py  [NEW]
├── minimization/
│   ├── __init__.py               [NEW]
│   └── _levenberg_marquardt.py   [NEW]
├── root_finding/
│   ├── __init__.py
│   └── _brent.py
└── test_functions/
    ├── __init__.py
    └── _rosenbrock.py
```

| Submodule | Operators | Implementation | Status |
|-----------|-----------|----------------|--------|
| `test_functions/` | rosenbrock | C++ | Done |
| `root_finding/` | brent | C++ | Done |
| `minimization/` | levenberg_marquardt | Python | NEW |
| `combinatorial/` | sinkhorn | C++ | NEW |
| `constrained/` | augmented_lagrangian | Python | NEW |

**Key patterns established:**

1. **Python-only solvers** (levenberg_marquardt, augmented_lagrangian): Use `torch.func.jacobian` for derivatives and `torch.autograd.Function` for implicit differentiation through the optimum.

2. **C++ tensor operators** (sinkhorn): Full C++ implementation with CPU, Meta, and Autograd backends following the rosenbrock pattern.

3. **Constrained optimization** (augmented_lagrangian): Equality and inequality constraints with implicit differentiation through KKT conditions.
