# Optimization Module Milestone 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish implementation patterns in the `minimization/` and `combinatorial/` submodules.

**Architecture:**
- `levenberg_marquardt`: Pure Python with implicit differentiation (requires function evaluation through autograd)
- `sinkhorn`: C++ implementation (operates on tensors only, no function evaluation)

**Tech Stack:** PyTorch, torch.func, torch.autograd.Function, ATen, TORCH_LIBRARY

---

## Milestone 1 Operators

| Submodule | Operator | Implementation | Pattern |
|-----------|----------|----------------|---------|
| `minimization/` | `levenberg_marquardt` | Pure Python | Iterative solver with implicit diff |
| `combinatorial/` | `sinkhorn` | C++ | Iterative, naturally differentiable |

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

## Summary

After completing both tasks, the optimization module structure is:

```
torchscience.optimization/
├── __init__.py
├── combinatorial/
│   ├── __init__.py
│   └── _sinkhorn.py          [NEW]
├── minimization/
│   ├── __init__.py           [NEW]
│   └── _levenberg_marquardt.py [NEW]
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
| `constrained/` | (none yet) | - | Milestone 2 |

**Key patterns established:**

1. **Python-only solvers** (levenberg_marquardt): Use `torch.func.jacobian` for derivatives and `torch.autograd.Function` for implicit differentiation through the optimum.

2. **C++ tensor operators** (sinkhorn): Full C++ implementation with CPU, Meta, and Autograd backends following the rosenbrock pattern.
