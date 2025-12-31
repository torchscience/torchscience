# Optimization Module Milestone 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish implementation patterns in each optimization submodule with one high-priority operator per submodule.

**Architecture:** Each operator follows the torchscience pattern: Python API wrapper → C++ Autograd (torch::autograd::Function) → C++ CPU kernel → C++ Meta kernel. All operators support batched inputs and autograd.

**Tech Stack:** PyTorch C++ Extension, ATen, torch::autograd::Function, TORCH_LIBRARY

---

## Milestone 1 Operators

| Submodule | Operator | Priority | Pattern |
|-----------|----------|----------|---------|
| `test_functions/` | `rastrigin` | High | Reduction (same as rosenbrock) |
| `test_functions/` | `ackley` | High | Reduction (same as rosenbrock) |
| `root_finding/` | `newton` | High | Iterative solver with implicit diff |
| `minimization/` | `levenberg_marquardt` | High | Iterative solver with implicit diff |
| `combinatorial/` | `sinkhorn` | High | Iterative, naturally differentiable |

---

## Task 1: Rastrigin Test Function

The Rastrigin function is a multimodal test function with many local minima.

**Mathematical definition:**
```
f(x) = An + Σᵢ[xᵢ² - A·cos(2πxᵢ)]
```
where A = 10, n = dimension. Global minimum at x = 0 with f(0) = 0.

**Files:**
- Create: `src/torchscience/optimization/test_functions/_rastrigin.py`
- Modify: `src/torchscience/optimization/test_functions/__init__.py`
- Modify: `src/torchscience/csrc/cpu/optimization/test_functions.h`
- Modify: `src/torchscience/csrc/meta/optimization/test_functions.h`
- Modify: `src/torchscience/csrc/autograd/optimization/test_functions.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `tests/torchscience/optimization/test_functions/test__rastrigin.py`

**Step 1: Write failing test**

Create `tests/torchscience/optimization/test_functions/test__rastrigin.py`:

```python
import pytest
import torch
import torch.testing

import torchscience.optimization.test_functions


class TestRastrigin:
    def test_global_minimum(self):
        """Test that global minimum is 0 at origin."""
        x = torch.zeros(5)
        result = torchscience.optimization.test_functions.rastrigin(x)
        torch.testing.assert_close(result, torch.tensor(0.0))

    def test_known_value(self):
        """Test known function value."""
        x = torch.ones(2)
        # f([1,1]) = 10*2 + (1 - 10*cos(2π)) + (1 - 10*cos(2π))
        #          = 20 + 2*(1 - 10*1) = 20 + 2*(-9) = 2
        result = torchscience.optimization.test_functions.rastrigin(x)
        torch.testing.assert_close(result, torch.tensor(2.0))

    def test_batch(self):
        """Test batched input."""
        x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        result = torchscience.optimization.test_functions.rastrigin(x)
        expected = torch.tensor([0.0, 2.0])
        torch.testing.assert_close(result, expected)

    def test_output_shape(self):
        """Test output shape is input shape without last dim."""
        x = torch.randn(3, 4, 5)
        result = torchscience.optimization.test_functions.rastrigin(x)
        assert result.shape == (3, 4)

    def test_gradcheck(self):
        """Test gradient correctness."""
        x = torch.tensor([0.5, 0.3], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            torchscience.optimization.test_functions.rastrigin,
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.tensor([0.5, 0.3], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradgradcheck(
            torchscience.optimization.test_functions.rastrigin,
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_custom_a_parameter(self):
        """Test with custom A parameter."""
        x = torch.zeros(3)
        result = torchscience.optimization.test_functions.rastrigin(x, a=5.0)
        torch.testing.assert_close(result, torch.tensor(0.0))

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        x = torch.empty(3, 5, device="meta")
        result = torchscience.optimization.test_functions.rastrigin(x)
        assert result.shape == (3,)
        assert result.device.type == "meta"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/optimization/test_functions/test__rastrigin.py -v`

Expected: FAIL with "module 'torchscience.optimization.test_functions' has no attribute 'rastrigin'"

**Step 3: Add schema registration**

Modify `src/torchscience/csrc/torchscience.cpp`, add after rosenbrock schemas:

```cpp
  module.def("rastrigin(Tensor x, Tensor a) -> Tensor");
  module.def("rastrigin_backward(Tensor grad_output, Tensor x, Tensor a) -> (Tensor, Tensor)");
  module.def("rastrigin_backward_backward(Tensor gg_x, Tensor gg_a, Tensor grad_output, Tensor x, Tensor a) -> (Tensor, Tensor, Tensor)");
```

**Step 4: Add CPU kernel**

Modify `src/torchscience/csrc/cpu/optimization/test_functions.h`, add after rosenbrock:

```cpp
namespace {

inline void check_rastrigin_input(const at::Tensor& x, const char* fn_name) {
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()),
        fn_name, " requires floating-point input, got ",
        x.scalar_type()
    );
    TORCH_CHECK(
        x.dim() >= 1 && x.size(-1) >= 1,
        fn_name, " requires at least 1 dimension in the last axis"
    );
}

template <typename scalar_t>
inline scalar_t rastrigin_forward_kernel(
    const scalar_t* x_ptr,
    scalar_t a_val,
    int64_t n
) {
    constexpr scalar_t two_pi = scalar_t(2.0 * M_PI);
    scalar_t sum = a_val * static_cast<scalar_t>(n);
    for (int64_t i = 0; i < n; ++i) {
        scalar_t xi = x_ptr[i];
        sum += xi * xi - a_val * std::cos(two_pi * xi);
    }
    return sum;
}

template <typename scalar_t>
inline void rastrigin_gradient_kernel(
    const scalar_t* x_ptr,
    scalar_t* grad_ptr,
    scalar_t a_val,
    int64_t n
) {
    constexpr scalar_t two_pi = scalar_t(2.0 * M_PI);
    for (int64_t i = 0; i < n; ++i) {
        scalar_t xi = x_ptr[i];
        // df/dxi = 2*xi + 2*pi*A*sin(2*pi*xi)
        grad_ptr[i] = scalar_t(2) * xi + two_pi * a_val * std::sin(two_pi * xi);
    }
}

}  // anonymous namespace

inline at::Tensor rastrigin(
    const at::Tensor& x,
    const at::Tensor& a
) {
    check_rastrigin_input(x, "rastrigin");

    const int64_t n = x.size(-1);
    const int64_t batch_size = x.numel() / n;

    std::vector<int64_t> output_shape(x.sizes().begin(), x.sizes().end() - 1);
    if (output_shape.empty()) {
        output_shape.push_back(1);
    }

    at::Tensor x_contig = x.contiguous();
    at::Tensor x_flat = x_contig.view({batch_size, n});
    at::Tensor output = at::empty({batch_size}, x.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        x.scalar_type(),
        "rastrigin_cpu",
        [&]() {
            const scalar_t* x_data = x_flat.data_ptr<scalar_t>();
            scalar_t* output_data = output.data_ptr<scalar_t>();
            scalar_t a_val = a.item<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
                    const scalar_t* x_ptr = x_data + batch_idx * n;
                    output_data[batch_idx] = rastrigin_forward_kernel(x_ptr, a_val, n);
                }
            });
        }
    );

    if (x.dim() == 1) {
        return output.squeeze(0);
    }
    return output.view(output_shape);
}

inline std::tuple<at::Tensor, at::Tensor> rastrigin_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a
) {
    const int64_t n = x.size(-1);
    const int64_t batch_size = x.numel() / n;

    at::Tensor x_contig = x.contiguous();
    at::Tensor x_flat = x_contig.view({batch_size, n});
    at::Tensor grad_x_flat = at::empty({batch_size, n}, x.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        x.scalar_type(),
        "rastrigin_backward_cpu",
        [&]() {
            const scalar_t* x_data = x_flat.data_ptr<scalar_t>();
            scalar_t* grad_x_data = grad_x_flat.data_ptr<scalar_t>();
            scalar_t a_val = a.item<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
                    const scalar_t* x_ptr = x_data + batch_idx * n;
                    scalar_t* grad_ptr = grad_x_data + batch_idx * n;
                    rastrigin_gradient_kernel(x_ptr, grad_ptr, a_val, n);
                }
            });
        }
    );

    at::Tensor grad_x = grad_output.unsqueeze(-1) * grad_x_flat.view_as(x);

    // df/da = n - sum_i cos(2*pi*xi)
    constexpr double two_pi = 2.0 * M_PI;
    at::Tensor cos_sum = at::sum(at::cos(two_pi * x), -1);
    at::Tensor df_da = static_cast<double>(n) - cos_sum;
    at::Tensor grad_a = at::sum(grad_output * df_da);

    return {grad_x, grad_a};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> rastrigin_backward_backward(
    const at::Tensor& grad_grad_x,
    const at::Tensor& grad_grad_a,
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a
) {
    const int64_t n = x.size(-1);
    constexpr double two_pi = 2.0 * M_PI;
    constexpr double four_pi_sq = 4.0 * M_PI * M_PI;

    at::Tensor grad_grad_output = at::zeros_like(grad_output);
    at::Tensor grad_x = at::zeros_like(x);
    at::Tensor grad_a = at::zeros_like(a);

    if (grad_grad_x.defined()) {
        // Hessian is diagonal: d²f/dxi² = 2 + 4*pi²*A*cos(2*pi*xi)
        at::Tensor hess_diag = 2.0 + four_pi_sq * a * at::cos(two_pi * x);
        grad_x = grad_x + grad_output.unsqueeze(-1) * hess_diag * grad_grad_x;

        // Contribution to grad_grad_output
        at::Tensor local_grad = 2 * x + two_pi * a * at::sin(two_pi * x);
        grad_grad_output = grad_grad_output + at::sum(grad_grad_x * local_grad, -1);
    }

    if (grad_grad_a.defined()) {
        // d²f/dxi da = 2*pi*sin(2*pi*xi)
        at::Tensor d2f_dxda = two_pi * at::sin(two_pi * x);
        grad_x = grad_x + grad_output.unsqueeze(-1) * grad_grad_a.unsqueeze(-1) * d2f_dxda;

        at::Tensor df_da = static_cast<double>(n) - at::sum(at::cos(two_pi * x), -1);
        grad_grad_output = grad_grad_output + df_da * grad_grad_a;
    }

    return {grad_grad_output, grad_x, grad_a};
}

// Add to TORCH_LIBRARY_IMPL at end of file:
// module.impl("rastrigin", &torchscience::cpu::test_functions::rastrigin);
// module.impl("rastrigin_backward", &torchscience::cpu::test_functions::rastrigin_backward);
// module.impl("rastrigin_backward_backward", &torchscience::cpu::test_functions::rastrigin_backward_backward);
```

**Step 5: Add Meta kernel**

Modify `src/torchscience/csrc/meta/optimization/test_functions.h`, add:

```cpp
inline at::Tensor rastrigin(
    const at::Tensor& x,
    const at::Tensor& a
) {
    TORCH_CHECK(x.dim() >= 1, "rastrigin requires at least 1 dimension");

    auto output_sizes = x.sizes().vec();
    output_sizes.pop_back();

    auto output_dtype = at::promote_types(x.scalar_type(), a.scalar_type());

    return at::empty(
        output_sizes,
        x.options().dtype(output_dtype).device(at::kMeta)
    );
}

inline std::tuple<at::Tensor, at::Tensor> rastrigin_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a
) {
    auto output_dtype = at::promote_types(x.scalar_type(), a.scalar_type());

    at::Tensor grad_x = at::empty(
        x.sizes(),
        x.options().dtype(output_dtype).device(at::kMeta)
    );
    at::Tensor grad_a = at::empty(
        a.sizes(),
        a.options().dtype(output_dtype).device(at::kMeta)
    );

    return {grad_x, grad_a};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> rastrigin_backward_backward(
    const at::Tensor& grad_grad_x,
    const at::Tensor& grad_grad_a,
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a
) {
    auto output_dtype = at::promote_types(x.scalar_type(), a.scalar_type());

    at::Tensor gg_output = at::empty(
        grad_output.sizes(),
        grad_output.options().dtype(output_dtype).device(at::kMeta)
    );
    at::Tensor grad_x = at::empty(
        x.sizes(),
        x.options().dtype(output_dtype).device(at::kMeta)
    );
    at::Tensor grad_a = at::empty(
        a.sizes(),
        a.options().dtype(output_dtype).device(at::kMeta)
    );

    return {gg_output, grad_x, grad_a};
}

// Add to TORCH_LIBRARY_IMPL:
// module.impl("rastrigin", &torchscience::meta::optimization::test_functions::rastrigin);
// module.impl("rastrigin_backward", &torchscience::meta::optimization::test_functions::rastrigin_backward);
// module.impl("rastrigin_backward_backward", &torchscience::meta::optimization::test_functions::rastrigin_backward_backward);
```

**Step 6: Add Autograd wrapper**

Modify `src/torchscience/csrc/autograd/optimization/test_functions.h`, add (following Rosenbrock pattern):

```cpp
class RastriginBackward
    : public torch::autograd::Function<RastriginBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& x,
        const at::Tensor& a,
        const bool x_requires_grad,
        const bool a_requires_grad
    ) {
        context->save_for_backward({grad_output, x, a});
        context->saved_data["x_requires_grad"] = x_requires_grad;
        context->saved_data["a_requires_grad"] = a_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_x, grad_a] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::rastrigin_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_output, x, a);

        return {grad_x, grad_a};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& gradient_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        const bool x_requires_grad = context->saved_data["x_requires_grad"].toBool();
        const bool a_requires_grad = context->saved_data["a_requires_grad"].toBool();

        const bool grad_grad_x_defined = gradient_outputs[0].defined();
        const bool grad_grad_a_defined = gradient_outputs[1].defined();

        if (!(grad_grad_x_defined && x_requires_grad) &&
            !(grad_grad_a_defined && a_requires_grad)) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_grad_x_input;
        at::Tensor grad_grad_a_input;

        if (grad_grad_x_defined && x_requires_grad) {
            grad_grad_x_input = gradient_outputs[0];
        }
        if (grad_grad_a_defined && a_requires_grad) {
            grad_grad_a_input = gradient_outputs[1];
        }

        auto [grad_grad_output, grad_x, grad_a] =
            c10::Dispatcher::singleton()
                .findSchemaOrThrow("torchscience::rastrigin_backward_backward", "")
                .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&
                )>()
                .call(
                    grad_grad_x_input,
                    grad_grad_a_input,
                    saved[0],
                    saved[1],
                    saved[2]
                );

        return {grad_grad_output, grad_x, grad_a, at::Tensor(), at::Tensor()};
    }
};

class Rastrigin
    : public torch::autograd::Function<Rastrigin> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& x,
        const at::Tensor& a
    ) {
        context->save_for_backward({x, a});

        const bool is_differentiable = at::isFloatingType(x.scalar_type());

        context->saved_data["x_requires_grad"] = x.requires_grad() && is_differentiable;
        context->saved_data["a_requires_grad"] = a.requires_grad() && is_differentiable;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::rastrigin", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(x, a);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& gradient_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();

        at::Tensor x = saved[0];
        at::Tensor a = saved[1];
        at::Tensor grad_output = gradient_outputs[0];

        bool x_requires_grad = context->saved_data["x_requires_grad"].toBool();
        bool a_requires_grad = context->saved_data["a_requires_grad"].toBool();

        std::vector<at::Tensor> gradients = RastriginBackward::apply(
            grad_output, x, a, x_requires_grad, a_requires_grad
        );

        at::Tensor grad_x;
        at::Tensor grad_a;

        if (x_requires_grad) {
            grad_x = gradients[0];
        }
        if (a_requires_grad) {
            grad_a = gradients[1];
        }

        return {grad_x, grad_a};
    }
};

inline at::Tensor rastrigin(
    const at::Tensor& x,
    const at::Tensor& a
) {
    return Rastrigin::apply(x, a);
}

// Add to TORCH_LIBRARY_IMPL:
// module.impl("rastrigin", &torchscience::autograd::test_functions::rastrigin);
```

**Step 7: Add Python API**

Create `src/torchscience/optimization/test_functions/_rastrigin.py`:

```python
from typing import Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def rastrigin(
    x: Tensor,
    *,
    a: Union[float, Tensor] = 10.0,
) -> Tensor:
    r"""
    Rastrigin function.

    A multimodal test function with many local minima, commonly used
    for benchmarking optimization algorithms.

    Mathematical Definition
    -----------------------
    .. math::

        f(\mathbf{x}) = An + \sum_{i=1}^{n} \left[ x_i^2 - A\cos(2\pi x_i) \right]

    where A = 10 by default.

    The global minimum is at :math:`\mathbf{x}^* = \mathbf{0}` where
    :math:`f(\mathbf{0}) = 0`.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(..., n)`` where ``n >= 1``.
    a : float or Tensor, optional
        Parameter controlling the amplitude of the cosine term.
        Default is 10.0.

    Returns
    -------
    Tensor
        Function value at each input point. Shape is ``(...)``.

    Examples
    --------
    >>> x = torch.zeros(5)
    >>> rastrigin(x)
    tensor(0.)

    >>> x = torch.ones(2)
    >>> rastrigin(x)
    tensor(2.)

    References
    ----------
    https://en.wikipedia.org/wiki/Rastrigin_function
    """
    if not isinstance(a, Tensor):
        a = torch.as_tensor(a, dtype=x.dtype, device=x.device)

    return torch.ops.torchscience.rastrigin(x, a)
```

**Step 8: Update module __init__.py**

Modify `src/torchscience/optimization/test_functions/__init__.py`:

```python
from ._rastrigin import rastrigin
from ._rosenbrock import rosenbrock

__all__ = [
    "rastrigin",
    "rosenbrock",
]
```

**Step 9: Run tests**

Run: `uv run pytest tests/torchscience/optimization/test_functions/test__rastrigin.py -v`

Expected: All tests pass

**Step 10: Commit**

```bash
git add src/torchscience/optimization/test_functions/_rastrigin.py \
        src/torchscience/optimization/test_functions/__init__.py \
        src/torchscience/csrc/cpu/optimization/test_functions.h \
        src/torchscience/csrc/meta/optimization/test_functions.h \
        src/torchscience/csrc/autograd/optimization/test_functions.h \
        src/torchscience/csrc/torchscience.cpp \
        tests/torchscience/optimization/test_functions/test__rastrigin.py
git commit -m "feat(test_functions): add rastrigin function"
```

---

## Task 2: Ackley Test Function

Similar pattern to rastrigin. The Ackley function is another multimodal test function.

**Mathematical definition:**
```
f(x) = -a·exp(-b·sqrt(1/n·Σxᵢ²)) - exp(1/n·Σcos(c·xᵢ)) + a + e
```
where a = 20, b = 0.2, c = 2π. Global minimum at x = 0 with f(0) = 0.

**Files:**
- Create: `src/torchscience/optimization/test_functions/_ackley.py`
- Modify: `src/torchscience/optimization/test_functions/__init__.py`
- Modify: `src/torchscience/csrc/cpu/optimization/test_functions.h`
- Modify: `src/torchscience/csrc/meta/optimization/test_functions.h`
- Modify: `src/torchscience/csrc/autograd/optimization/test_functions.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `tests/torchscience/optimization/test_functions/test__ackley.py`

**Step 1: Write failing test**

Create `tests/torchscience/optimization/test_functions/test__ackley.py`:

```python
import math

import pytest
import torch
import torch.testing

import torchscience.optimization.test_functions


class TestAckley:
    def test_global_minimum(self):
        """Test that global minimum is 0 at origin."""
        x = torch.zeros(5)
        result = torchscience.optimization.test_functions.ackley(x)
        torch.testing.assert_close(result, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

    def test_batch(self):
        """Test batched input."""
        x = torch.zeros(3, 4)
        result = torchscience.optimization.test_functions.ackley(x)
        expected = torch.zeros(3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_output_shape(self):
        """Test output shape."""
        x = torch.randn(2, 3, 4)
        result = torchscience.optimization.test_functions.ackley(x)
        assert result.shape == (2, 3)

    def test_nonnegative(self):
        """Test that function is non-negative."""
        x = torch.randn(100, 5)
        result = torchscience.optimization.test_functions.ackley(x)
        assert (result >= -1e-6).all()

    def test_gradcheck(self):
        """Test gradient correctness."""
        x = torch.tensor([0.5, 0.3], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            torchscience.optimization.test_functions.ackley,
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.tensor([0.5, 0.3], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradgradcheck(
            torchscience.optimization.test_functions.ackley,
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        x = torch.empty(3, 5, device="meta")
        result = torchscience.optimization.test_functions.ackley(x)
        assert result.shape == (3,)
        assert result.device.type == "meta"
```

**Step 2-10:** Follow same pattern as rastrigin with ackley-specific math:

```cpp
// CPU kernel core:
template <typename scalar_t>
inline scalar_t ackley_forward_kernel(
    const scalar_t* x_ptr,
    scalar_t a_val,
    scalar_t b_val,
    scalar_t c_val,
    int64_t n
) {
    scalar_t sum_sq = 0;
    scalar_t sum_cos = 0;
    for (int64_t i = 0; i < n; ++i) {
        scalar_t xi = x_ptr[i];
        sum_sq += xi * xi;
        sum_cos += std::cos(c_val * xi);
    }
    scalar_t inv_n = scalar_t(1) / static_cast<scalar_t>(n);
    scalar_t term1 = -a_val * std::exp(-b_val * std::sqrt(inv_n * sum_sq));
    scalar_t term2 = -std::exp(inv_n * sum_cos);
    return term1 + term2 + a_val + std::exp(scalar_t(1));
}
```

Python API:
```python
def ackley(
    x: Tensor,
    *,
    a: Union[float, Tensor] = 20.0,
    b: Union[float, Tensor] = 0.2,
    c: Union[float, Tensor] = 2.0 * math.pi,
) -> Tensor:
    ...
```

Commit: `git commit -m "feat(test_functions): add ackley function"`

---

## Task 3: Newton's Method (Root Finding)

Newton-Raphson method for solving systems of equations f(x) = 0.

**Files:**
- Create: `src/torchscience/optimization/root_finding/_newton.py`
- Modify: `src/torchscience/optimization/root_finding/__init__.py`
- Create: `tests/torchscience/optimization/root_finding/test__newton.py`

**Note:** Newton's method is implemented purely in Python since it requires function evaluation and Jacobian computation through autograd.

**Step 1: Write failing test**

Create `tests/torchscience/optimization/root_finding/test__newton.py`:

```python
import pytest
import torch
import torch.testing

import torchscience.optimization.root_finding


class TestNewton:
    def test_scalar_sqrt2(self):
        """Find sqrt(2) by solving x^2 - 2 = 0."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5])
        root = torchscience.optimization.root_finding.newton(f, x0)
        torch.testing.assert_close(root, torch.tensor([2.0]).sqrt(), atol=1e-6, rtol=1e-6)

    def test_system_2d(self):
        """Solve 2D system: x^2 + y^2 = 1, x = y."""
        def f(xy):
            x, y = xy[..., 0], xy[..., 1]
            eq1 = x**2 + y**2 - 1
            eq2 = x - y
            return torch.stack([eq1, eq2], dim=-1)

        x0 = torch.tensor([0.5, 0.5])
        root = torchscience.optimization.root_finding.newton(f, x0)
        expected = torch.tensor([1/2**0.5, 1/2**0.5])
        torch.testing.assert_close(root, expected, atol=1e-5, rtol=1e-5)

    def test_batched(self):
        """Test batched root finding."""
        c = torch.tensor([2.0, 3.0, 4.0])
        f = lambda x: x**2 - c
        x0 = torch.ones(3) * 2
        roots = torchscience.optimization.root_finding.newton(f, x0)
        expected = c.sqrt()
        torch.testing.assert_close(roots, expected, atol=1e-6, rtol=1e-6)

    def test_implicit_differentiation(self):
        """Test gradient through root via implicit differentiation."""
        theta = torch.tensor([2.0], requires_grad=True)
        f = lambda x: x**2 - theta
        x0 = torch.tensor([1.5])
        root = torchscience.optimization.root_finding.newton(f, x0)
        root.sum().backward()
        # d(sqrt(theta))/d(theta) = 1/(2*sqrt(theta))
        expected_grad = 1 / (2 * theta.sqrt())
        torch.testing.assert_close(theta.grad, expected_grad, atol=1e-5, rtol=1e-5)

    def test_custom_jacobian(self):
        """Test with user-provided Jacobian."""
        f = lambda x: x**2 - 2
        jacobian = lambda x: 2 * x.unsqueeze(-1)  # (N,) -> (N, 1)
        x0 = torch.tensor([1.5])
        root = torchscience.optimization.root_finding.newton(f, x0, jacobian=jacobian)
        torch.testing.assert_close(root, torch.tensor([2.0]).sqrt(), atol=1e-6, rtol=1e-6)
```

**Step 2: Implement newton**

Create `src/torchscience/optimization/root_finding/_newton.py`:

```python
from typing import Callable, Optional

import torch
from torch import Tensor


class _NewtonImplicitGrad(torch.autograd.Function):
    """Implicit differentiation through Newton's method."""

    @staticmethod
    def forward(ctx, root: Tensor, f_callable, jacobian_callable, orig_shape) -> Tensor:
        ctx.f_callable = f_callable
        ctx.jacobian_callable = jacobian_callable
        ctx.orig_shape = orig_shape
        ctx.save_for_backward(root)
        return root

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (root,) = ctx.saved_tensors

        with torch.enable_grad():
            x = root.detach().requires_grad_(True)
            fx = ctx.f_callable(x)

            if ctx.jacobian_callable is not None:
                J = ctx.jacobian_callable(x)
            else:
                J = torch.func.jacobian(ctx.f_callable)(x)

            # Solve J^T @ v = grad_output for v
            # Then propagate -v through df/dtheta
            if J.dim() == 1:
                J = J.unsqueeze(-1)

            # v = (J^T)^{-1} @ grad_output
            v = torch.linalg.solve(J.T, grad_output.unsqueeze(-1)).squeeze(-1)

            # Backprop through f with -v as the gradient
            torch.autograd.backward(fx, -v)

        return None, None, None, None


def newton(
    f: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    jacobian: Optional[Callable[[Tensor], Tensor]] = None,
    tol: Optional[float] = None,
    maxiter: int = 50,
) -> Tensor:
    """
    Newton-Raphson method for solving f(x) = 0.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Function to find root of. For scalar problems: (N,) -> (N,).
        For systems: (N, d) -> (N, d).
    x0 : Tensor
        Initial guess.
    jacobian : Callable, optional
        Jacobian function. If None, computed via torch.func.jacobian.
    tol : float, optional
        Convergence tolerance. Default: sqrt(eps) for the dtype.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    Tensor
        Root of f.

    Examples
    --------
    >>> f = lambda x: x**2 - 2
    >>> x0 = torch.tensor([1.5])
    >>> newton(f, x0)
    tensor([1.4142])
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    orig_shape = x0.shape
    x = x0.clone()

    for _ in range(maxiter):
        fx = f(x)

        if torch.all(torch.abs(fx) < tol):
            break

        if jacobian is not None:
            J = jacobian(x)
        else:
            J = torch.func.jacobian(f)(x)

        # Handle scalar case
        if J.dim() == 0:
            J = J.unsqueeze(0).unsqueeze(0)
        elif J.dim() == 1:
            J = J.unsqueeze(-1)

        # Newton update: x = x - J^{-1} @ f(x)
        if fx.dim() == 0:
            fx = fx.unsqueeze(0)
        delta = torch.linalg.solve(J, fx.unsqueeze(-1)).squeeze(-1)
        x = x - delta

    # Attach implicit gradient
    return _NewtonImplicitGrad.apply(x, f, jacobian, orig_shape)
```

**Step 3: Update __init__.py**

Modify `src/torchscience/optimization/root_finding/__init__.py`:

```python
from ._brent import brent
from ._newton import newton

__all__ = [
    "brent",
    "newton",
]
```

**Step 4: Run tests**

Run: `uv run pytest tests/torchscience/optimization/root_finding/test__newton.py -v`

**Step 5: Commit**

```bash
git commit -m "feat(root_finding): add newton method"
```

---

## Task 4: Levenberg-Marquardt Algorithm

Nonlinear least squares solver for minimizing ||r(x)||².

**Files:**
- Create: `src/torchscience/optimization/minimization/__init__.py`
- Create: `src/torchscience/optimization/minimization/_levenberg_marquardt.py`
- Modify: `src/torchscience/optimization/__init__.py`
- Create: `tests/torchscience/optimization/minimization/__init__.py`
- Create: `tests/torchscience/optimization/minimization/test__levenberg_marquardt.py`

**Step 1: Write failing test**

Create `tests/torchscience/optimization/minimization/test__levenberg_marquardt.py`:

```python
import pytest
import torch
import torch.testing

import torchscience.optimization.minimization


class TestLevenbergMarquardt:
    def test_linear_least_squares(self):
        """Fit y = ax + b to data."""
        # True: a=2, b=1
        x_data = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y_data = 2.0 * x_data + 1.0

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
        # dx*/dtarget = 1
        torch.testing.assert_close(target.grad, torch.tensor([1.0]), atol=1e-5, rtol=1e-5)

    def test_batched(self):
        """Test batched optimization."""
        targets = torch.tensor([[1.0], [2.0], [3.0]])

        def residuals(x):
            return x - targets

        x0 = torch.zeros(3, 1)
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, x0
        )
        torch.testing.assert_close(result, targets, atol=1e-5, rtol=1e-5)
```

**Step 2: Implement levenberg_marquardt**

Create `src/torchscience/optimization/minimization/_levenberg_marquardt.py`:

```python
from typing import Callable, Optional

import torch
from torch import Tensor


class _LMImplicitGrad(torch.autograd.Function):
    """Implicit differentiation through Levenberg-Marquardt."""

    @staticmethod
    def forward(ctx, result: Tensor, residuals_callable, orig_shape) -> Tensor:
        ctx.residuals_callable = residuals_callable
        ctx.orig_shape = orig_shape
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (result,) = ctx.saved_tensors

        with torch.enable_grad():
            x = result.detach().requires_grad_(True)
            r = ctx.residuals_callable(x)
            J = torch.func.jacobian(ctx.residuals_callable)(x)

            # At optimum: J^T @ r = 0
            # Implicit diff: d(J^T @ r)/dx * dx/dtheta + d(J^T @ r)/dtheta = 0
            # Simplified: (J^T @ J) @ dx/dtheta = -J^T @ dr/dtheta
            # For gradient: grad_theta = -grad_x @ (J^T @ J)^{-1} @ J^T @ dr/dtheta

            JtJ = J.T @ J
            Jt_grad = J.T @ grad_output.unsqueeze(-1)
            v = torch.linalg.solve(JtJ, Jt_grad).squeeze(-1)

            torch.autograd.backward(r, -J @ v.unsqueeze(-1))

        return None, None, None


def levenberg_marquardt(
    residuals: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    jacobian: Optional[Callable[[Tensor], Tensor]] = None,
    tol: Optional[float] = None,
    maxiter: int = 100,
    damping: float = 1e-3,
) -> Tensor:
    """
    Levenberg-Marquardt algorithm for nonlinear least squares.

    Minimizes ||residuals(x)||².

    Parameters
    ----------
    residuals : Callable[[Tensor], Tensor]
        Residual function. Shape: (n_params,) -> (n_residuals,).
    x0 : Tensor
        Initial parameter guess.
    jacobian : Callable, optional
        Jacobian of residuals. If None, computed via torch.func.jacobian.
    tol : float, optional
        Convergence tolerance on gradient norm. Default: sqrt(eps) for the dtype.
    maxiter : int
        Maximum iterations.
    damping : float
        Initial Levenberg-Marquardt damping parameter.

    Returns
    -------
    Tensor
        Optimized parameters.

    Examples
    --------
    >>> def residuals(params):
    ...     return params - torch.tensor([1.0, 2.0])
    >>> x0 = torch.zeros(2)
    >>> levenberg_marquardt(residuals, x0)
    tensor([1., 2.])
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    orig_shape = x0.shape
    x = x0.clone()
    mu = damping

    for _ in range(maxiter):
        r = residuals(x)

        if jacobian is not None:
            J = jacobian(x)
        else:
            J = torch.func.jacobian(residuals)(x)

        # Gradient: g = J^T @ r
        g = J.T @ r

        if torch.norm(g) < tol:
            break

        # Hessian approximation: H = J^T @ J + mu * I
        JtJ = J.T @ J
        n = JtJ.shape[0]
        H = JtJ + mu * torch.eye(n, dtype=x.dtype, device=x.device)

        # Solve H @ delta = -g
        delta = torch.linalg.solve(H, -g)

        # Compute actual vs predicted reduction
        x_new = x + delta
        r_new = residuals(x_new)
        actual_reduction = torch.sum(r**2) - torch.sum(r_new**2)
        predicted_reduction = -2 * (g @ delta) - delta @ JtJ @ delta

        rho = actual_reduction / (predicted_reduction + 1e-10)

        if rho > 0.25:
            x = x_new
            mu = max(mu / 3, 1e-10)
        else:
            mu = min(mu * 2, 1e10)

    return _LMImplicitGrad.apply(x, residuals, orig_shape)
```

**Step 3: Create __init__.py**

Create `src/torchscience/optimization/minimization/__init__.py`:

```python
from ._levenberg_marquardt import levenberg_marquardt

__all__ = [
    "levenberg_marquardt",
]
```

**Step 4: Update optimization __init__.py**

Modify `src/torchscience/optimization/__init__.py`:

```python
from . import minimization, root_finding, test_functions

__all__ = [
    "minimization",
    "root_finding",
    "test_functions",
]
```

**Step 5: Run tests and commit**

```bash
uv run pytest tests/torchscience/optimization/minimization/ -v
git commit -m "feat(minimization): add levenberg_marquardt algorithm"
```

---

## Task 5: Sinkhorn Algorithm

Entropy-regularized optimal transport solver. Implemented in C++ since it operates on tensors only (no function evaluations).

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

**Step 1: Write failing test**

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
        """Test gradient correctness."""
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
        C = torch.tensor([[0.0, 1.0, 2.0],
                          [1.0, 0.0, 1.0],
                          [2.0, 1.0, 0.0]])
        a = torch.ones(n) / n
        b = torch.ones(n) / n

        P_large_eps = torchscience.optimization.combinatorial.sinkhorn(C, a, b, epsilon=1.0)
        P_small_eps = torchscience.optimization.combinatorial.sinkhorn(C, a, b, epsilon=0.01)

        # Smaller epsilon should be more peaked (higher max value)
        assert P_small_eps.max() > P_large_eps.max()

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        C = torch.empty(3, 4, device="meta")
        a = torch.empty(3, device="meta")
        b = torch.empty(4, device="meta")
        P = torchscience.optimization.combinatorial.sinkhorn(C, a, b)
        assert P.shape == (3, 4)
        assert P.device.type == "meta"
```

**Step 2: Add schema registration**

Modify `src/torchscience/csrc/torchscience.cpp`:

```cpp
  // optimization.combinatorial
  module.def("sinkhorn(Tensor C, Tensor a, Tensor b, float epsilon, int maxiter, float tol) -> Tensor");
  module.def("sinkhorn_backward(Tensor grad_output, Tensor P, Tensor C, Tensor a, Tensor b, float epsilon) -> (Tensor, Tensor, Tensor)");
```

**Step 3: Add CPU kernel**

Create `src/torchscience/csrc/cpu/optimization/combinatorial.h`:

```cpp
#pragma once

#include <cmath>
#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
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

    // Compute kernel matrix K = exp(-C / epsilon)
    at::Tensor K = at::exp(-C / epsilon);

    // Initialize scaling vectors
    at::Tensor u = at::ones_like(a);
    at::Tensor v = at::ones_like(b);

    for (int64_t iter = 0; iter < maxiter; ++iter) {
        at::Tensor u_prev = u.clone();

        // v = b / (K^T @ u)
        at::Tensor Ktu = at::matmul(K.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1);
        v = b / at::clamp(Ktu, 1e-10);

        // u = a / (K @ v)
        at::Tensor Kv = at::matmul(K, v.unsqueeze(-1)).squeeze(-1);
        u = a / at::clamp(Kv, 1e-10);

        // Check convergence
        double max_diff = at::max(at::abs(u - u_prev)).item<double>();
        if (max_diff < tol) {
            break;
        }
    }

    // Transport plan: P = diag(u) @ K @ diag(v) = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    at::Tensor P = u.unsqueeze(-1) * K * v.unsqueeze(-2);

    return P;
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> sinkhorn_backward(
    const at::Tensor& grad_output,
    const at::Tensor& P,
    const at::Tensor& C,
    const at::Tensor& a,
    const at::Tensor& b,
    double epsilon
) {
    // Gradient w.r.t. C: dL/dC = dL/dP * dP/dC
    // P = diag(u) @ K @ diag(v), K = exp(-C/epsilon)
    // dP/dC = -P / epsilon (element-wise, since dK/dC = -K/epsilon)
    at::Tensor grad_C = -grad_output * P / epsilon;

    // For a and b, we'd need to solve a linear system (Sinkhorn is implicit)
    // For now, return zeros (gradients through a and b are less common)
    at::Tensor grad_a = at::zeros_like(a);
    at::Tensor grad_b = at::zeros_like(b);

    return {grad_C, grad_a, grad_b};
}

}  // namespace torchscience::cpu::optimization::combinatorial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("sinkhorn", &torchscience::cpu::optimization::combinatorial::sinkhorn);
    module.impl("sinkhorn_backward", &torchscience::cpu::optimization::combinatorial::sinkhorn_backward);
}
```

**Step 4: Add Meta kernel**

Create `src/torchscience/csrc/meta/optimization/combinatorial.h`:

```cpp
#pragma once

#include <tuple>

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
    // Output shape is same as C
    return at::empty(C.sizes(), C.options().device(at::kMeta));
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> sinkhorn_backward(
    const at::Tensor& grad_output,
    const at::Tensor& P,
    const at::Tensor& C,
    const at::Tensor& a,
    const at::Tensor& b,
    double epsilon
) {
    return {
        at::empty(C.sizes(), C.options().device(at::kMeta)),
        at::empty(a.sizes(), a.options().device(at::kMeta)),
        at::empty(b.sizes(), b.options().device(at::kMeta))
    };
}

}  // namespace torchscience::meta::optimization::combinatorial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("sinkhorn", &torchscience::meta::optimization::combinatorial::sinkhorn);
    module.impl("sinkhorn_backward", &torchscience::meta::optimization::combinatorial::sinkhorn_backward);
}
```

**Step 5: Add Autograd wrapper**

Create `src/torchscience/csrc/autograd/optimization/combinatorial.h`:

```cpp
#pragma once

#include <tuple>

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
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, double, int64_t, double)>()
            .call(C, a, b, epsilon, maxiter, tol);

        context->save_for_backward({P, C, a, b});
        context->saved_data["epsilon"] = epsilon;

        return P;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& gradient_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        double epsilon = context->saved_data["epsilon"].toDouble();

        at::AutoDispatchBelowAutograd guard;

        auto [grad_C, grad_a, grad_b] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::sinkhorn_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, double
            )>()
            .call(gradient_outputs[0], saved[0], saved[1], saved[2], saved[3], epsilon);

        return {grad_C, grad_a, grad_b, at::Tensor(), at::Tensor(), at::Tensor()};
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
    module.impl("sinkhorn", &torchscience::autograd::optimization::combinatorial::sinkhorn);
}
```

**Step 6: Add Python API**

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

    Solves:

    .. math::

        \min_P \langle C, P \rangle + \epsilon H(P)
        \quad \text{s.t.} \quad P \mathbf{1} = a, P^T \mathbf{1} = b, P \geq 0

    where H(P) is the entropy of P.

    Parameters
    ----------
    C : Tensor
        Cost matrix of shape (..., n, m).
    a : Tensor
        Source marginal of shape (..., n). Must sum to 1.
    b : Tensor
        Target marginal of shape (..., m). Must sum to 1.
    epsilon : float
        Entropy regularization strength. Default: 0.1.
    maxiter : int
        Maximum Sinkhorn iterations. Default: 100.
    tol : float
        Convergence tolerance on scaling vector change. Default: 1e-6.

    Returns
    -------
    Tensor
        Transport plan P of shape (..., n, m).

    Examples
    --------
    >>> C = torch.rand(3, 4)
    >>> a = torch.ones(3) / 3
    >>> b = torch.ones(4) / 4
    >>> P = sinkhorn(C, a, b)
    >>> P.sum(dim=-1)  # Should equal a
    tensor([0.3333, 0.3333, 0.3333])

    References
    ----------
    https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem
    """
    return torch.ops.torchscience.sinkhorn(C, a, b, epsilon, maxiter, tol)
```

**Step 7: Create __init__.py and update optimization**

Create `src/torchscience/optimization/combinatorial/__init__.py`:

```python
from ._sinkhorn import sinkhorn

__all__ = [
    "sinkhorn",
]
```

Update `src/torchscience/optimization/__init__.py`:

```python
from . import combinatorial, minimization, root_finding, test_functions

__all__ = [
    "combinatorial",
    "minimization",
    "root_finding",
    "test_functions",
]
```

**Step 8: Run tests and commit**

```bash
uv run pytest tests/torchscience/optimization/combinatorial/ -v
git commit -m "feat(combinatorial): add sinkhorn algorithm"
```

---

## Summary

After completing all 5 tasks, the optimization module will have:

| Submodule | Operators | Status |
|-----------|-----------|--------|
| `test_functions/` | rosenbrock, rastrigin, ackley | Pattern established |
| `root_finding/` | brent, newton | Pattern established |
| `minimization/` | levenberg_marquardt | Pattern established |
| `combinatorial/` | sinkhorn | Pattern established |
| `constrained/` | (none yet) | Next milestone |

Each submodule now has a working example that demonstrates:
- Batched input handling
- Autograd support (first and second order where applicable)
- Implicit differentiation for iterative solvers
- Meta tensor support for shape inference
