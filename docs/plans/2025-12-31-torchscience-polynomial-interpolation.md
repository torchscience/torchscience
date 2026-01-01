# torchscience.polynomial Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement differentiable polynomial interpolation module for torchscience, including classical methods and Chebyshev polynomials.

**Architecture:** Each operator follows the established pattern: Python API -> C++ schema registration -> CPU kernel + Meta implementation + Autograd wrapper. All operators support batched inputs and full autograd.

**Tech Stack:** PyTorch C++ extensions, TORCH_LIBRARY for schema, AT_DISPATCH_FLOATING_TYPES for kernels.

---

## Module Structure

```
torchscience/
└── polynomial/                        # Polynomial interpolation (incl. Chebyshev)
    ├── __init__.py
    ├── _lagrange_polynomial.py        # Task 1: Lagrange polynomial
    ├── _newton_polynomial.py          # Task 2: Newton divided differences
    ├── _barycentric_interpolation.py  # Task 3: Barycentric rational
    ├── _chebyshev_nodes.py            # Task 4: Chebyshev node generation
    └── _chebyshev_interpolation.py    # Task 5: Chebyshev polynomial interp
```

---

## Summary

| Operator | Wikipedia Article | Task |
|----------|-------------------|------|
| `lagrange_polynomial` | [Lagrange polynomial](https://en.wikipedia.org/wiki/Lagrange_polynomial) | Task 1 |
| `newton_polynomial` | [Newton polynomial](https://en.wikipedia.org/wiki/Newton_polynomial) | Task 2 |
| `barycentric_interpolation` | [Polynomial interpolation](https://en.wikipedia.org/wiki/Polynomial_interpolation) | Task 3 |
| `chebyshev_nodes` | [Chebyshev nodes](https://en.wikipedia.org/wiki/Chebyshev_nodes) | Task 4 |
| `chebyshev_interpolation` | [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) | Task 5 |

---

### Task 1: Add `lagrange_polynomial` interpolation operator

**Goal:** Implement Lagrange polynomial interpolation.

**Mathematical Definition:**
$$L(x) = \sum_{j=0}^{n} y_j \prod_{k \neq j} \frac{x - x_k}{x_j - x_k}$$

The Lagrange basis polynomials are:
$$\ell_j(x) = \prod_{k \neq j} \frac{x - x_k}{x_j - x_k}$$

**Files:**
- Create: `tests/torchscience/polynomial/__init__.py`
- Create: `tests/torchscience/polynomial/test__lagrange_polynomial.py`
- Create: `src/torchscience/csrc/kernel/polynomial/lagrange_polynomial.h`
- Create: `src/torchscience/csrc/kernel/polynomial/lagrange_polynomial_backward.h`
- Create: `src/torchscience/csrc/cpu/polynomial/lagrange_polynomial.h`
- Create: `src/torchscience/csrc/meta/polynomial/lagrange_polynomial.h`
- Create: `src/torchscience/csrc/autograd/polynomial/lagrange_polynomial.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/polynomial/__init__.py`
- Create: `src/torchscience/polynomial/_lagrange_polynomial.py`

#### Step 1.1: Write the failing test

Create `tests/torchscience/polynomial/__init__.py` (empty file).

Create `tests/torchscience/polynomial/test__lagrange_polynomial.py`:

```python
"""Tests for Lagrange polynomial interpolation."""

import math
import pytest
import torch
from torch.autograd import gradcheck


class TestLagrangePolynomialBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_matches_query(self):
        """Output shape matches query points shape."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.linspace(0, 1, 5)
        y = torch.sin(2 * math.pi * x)
        query = torch.linspace(0, 1, 50)

        result = lagrange_polynomial(x, y, query)

        assert result.shape == query.shape

    def test_output_shape_batch(self):
        """Output shape matches batch dimensions."""
        from torchscience.polynomial import lagrange_polynomial

        batch = 5
        n_points = 6
        n_query = 20

        x = torch.linspace(0, 1, n_points).unsqueeze(0).expand(batch, -1)
        y = torch.randn(batch, n_points)
        query = torch.linspace(0, 1, n_query).unsqueeze(0).expand(batch, -1)

        result = lagrange_polynomial(x, y, query)

        assert result.shape == (batch, n_query)

    def test_interpolation_at_nodes(self):
        """Interpolation exactly matches data at node points."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        y = torch.tensor([1.0, 2.0, 0.5, 3.0, 2.5], dtype=torch.float64)

        result = lagrange_polynomial(x, y, x)

        torch.testing.assert_close(result, y, rtol=1e-10, atol=1e-10)


class TestLagrangePolynomialCorrectness:
    """Tests for numerical correctness."""

    def test_linear_data_exact(self):
        """Lagrange interpolation reproduces linear functions exactly."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        y = 2 * x + 3  # Linear function
        query = torch.linspace(0, 1, 101, dtype=torch.float64)

        result = lagrange_polynomial(x, y, query)
        expected = 2 * query + 3

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_quadratic_data_exact(self):
        """Lagrange interpolation with 3 points reproduces quadratic exactly."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        y = x ** 2  # Quadratic function
        query = torch.linspace(-1, 1, 101, dtype=torch.float64)

        result = lagrange_polynomial(x, y, query)
        expected = query ** 2

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_cubic_data_exact(self):
        """Lagrange interpolation with 4 points reproduces cubic exactly."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.tensor([-1.0, 0.0, 0.5, 1.0], dtype=torch.float64)
        y = x ** 3 - 2 * x + 1  # Cubic function
        query = torch.linspace(-1, 1, 101, dtype=torch.float64)

        result = lagrange_polynomial(x, y, query)
        expected = query ** 3 - 2 * query + 1

        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_single_point(self):
        """Interpolation with single point returns constant."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.tensor([0.5], dtype=torch.float64)
        y = torch.tensor([3.0], dtype=torch.float64)
        query = torch.linspace(0, 1, 10, dtype=torch.float64)

        result = lagrange_polynomial(x, y, query)
        expected = torch.full_like(query, 3.0)

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)


class TestLagrangePolynomialGradients:
    """Tests for gradient computation."""

    def test_gradcheck_y(self):
        """Passes gradcheck for y values."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.randn(5, dtype=torch.float64, requires_grad=True)
        query = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)

        def func(y_):
            return lagrange_polynomial(x, y_, query)

        assert gradcheck(func, (y,), raise_exception=True)

    def test_gradcheck_query(self):
        """Passes gradcheck for query points."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.randn(5, dtype=torch.float64)
        query = torch.linspace(0.1, 0.9, 5, dtype=torch.float64, requires_grad=True)

        def func(q):
            return lagrange_polynomial(x, y, q)

        assert gradcheck(func, (query,), raise_exception=True)

    def test_gradients_finite(self):
        """Gradients are finite for typical inputs."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.randn(5, dtype=torch.float64, requires_grad=True)
        query = torch.linspace(0.1, 0.9, 20, dtype=torch.float64)

        result = lagrange_polynomial(x, y, query)
        result.sum().backward()

        assert y.grad is not None
        assert torch.isfinite(y.grad).all()


class TestLagrangePolynomialEdgeCases:
    """Tests for edge cases."""

    def test_two_points(self):
        """Works with two points (linear interpolation)."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        y = torch.tensor([1.0, 3.0], dtype=torch.float64)
        query = torch.tensor([0.5], dtype=torch.float64)

        result = lagrange_polynomial(x, y, query)

        torch.testing.assert_close(result, torch.tensor([2.0], dtype=torch.float64))

    def test_duplicate_x_raises(self):
        """Raises error when x values have duplicates."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.tensor([0.0, 1.0, 1.0, 2.0])  # Duplicate
        y = torch.tensor([1.0, 2.0, 2.0, 1.5])
        query = torch.tensor([0.5])

        with pytest.raises(ValueError, match="x values must be distinct"):
            lagrange_polynomial(x, y, query)


class TestLagrangePolynomialDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.linspace(0, 1, 5, dtype=torch.float32)
        y = torch.randn(5, dtype=torch.float32)
        query = torch.linspace(0, 1, 20, dtype=torch.float32)

        result = lagrange_polynomial(x, y, query)

        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.polynomial import lagrange_polynomial

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.randn(5, dtype=torch.float64)
        query = torch.linspace(0, 1, 20, dtype=torch.float64)

        result = lagrange_polynomial(x, y, query)

        assert result.dtype == torch.float64


class TestLagrangePolynomialRungePhenomenon:
    """Tests demonstrating Runge phenomenon for awareness."""

    def test_runge_function_few_points(self):
        """Demonstrates reasonable behavior with few uniform points."""
        from torchscience.polynomial import lagrange_polynomial

        # Runge function: 1 / (1 + 25*x^2)
        x = torch.linspace(-1, 1, 5, dtype=torch.float64)
        y = 1 / (1 + 25 * x ** 2)
        query = torch.linspace(-1, 1, 100, dtype=torch.float64)

        result = lagrange_polynomial(x, y, query)

        # With 5 points, should be reasonably accurate
        assert torch.isfinite(result).all()

    def test_runge_function_many_points_oscillation(self):
        """More uniform points can cause oscillation (Runge phenomenon)."""
        from torchscience.polynomial import lagrange_polynomial

        # With many uniform points, Lagrange can oscillate wildly at edges
        x = torch.linspace(-1, 1, 15, dtype=torch.float64)
        y = 1 / (1 + 25 * x ** 2)
        query = torch.linspace(-1, 1, 100, dtype=torch.float64)

        result = lagrange_polynomial(x, y, query)

        # Result should still be finite, but may oscillate
        assert torch.isfinite(result).all()
        # Note: Max absolute value may be large due to Runge phenomenon
```

#### Step 1.2: Run test to verify it fails

Run: `uv run pytest tests/torchscience/polynomial/test__lagrange_polynomial.py -v`
Expected: FAIL with "No module named 'torchscience.polynomial'"

#### Step 1.3: Create kernel header (forward)

Create `src/torchscience/csrc/kernel/polynomial/lagrange_polynomial.h`:

```cpp
#pragma once

#include <cmath>

namespace torchscience::kernel::polynomial {

// Evaluate Lagrange basis polynomial ℓ_j(x)
// ℓ_j(x) = ∏_{k≠j} (x - x_k) / (x_j - x_k)
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T lagrange_basis(
    const T* x,      // node x-coordinates (n points)
    int64_t j,       // basis index
    T query,         // query point
    int64_t n
) {
    T result = T(1);
    for (int64_t k = 0; k < n; ++k) {
        if (k != j) {
            result *= (query - x[k]) / (x[j] - x[k]);
        }
    }
    return result;
}

// Evaluate Lagrange polynomial at query point
// L(x) = Σ_j y_j * ℓ_j(x)
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T evaluate_lagrange_polynomial(
    const T* x,      // node x-coordinates (n points)
    const T* y,      // node y-values (n points)
    T query,         // query point
    int64_t n
) {
    T result = T(0);
    for (int64_t j = 0; j < n; ++j) {
        result += y[j] * lagrange_basis(x, j, query, n);
    }
    return result;
}

// Derivative of Lagrange basis polynomial ℓ_j(x)
// Using product rule: d/dx [∏_{k≠j} (x - x_k)/(x_j - x_k)]
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T lagrange_basis_derivative(
    const T* x,      // node x-coordinates (n points)
    int64_t j,       // basis index
    T query,         // query point
    int64_t n
) {
    T result = T(0);
    for (int64_t i = 0; i < n; ++i) {
        if (i == j) continue;

        T term = T(1) / (x[j] - x[i]);
        for (int64_t k = 0; k < n; ++k) {
            if (k != j && k != i) {
                term *= (query - x[k]) / (x[j] - x[k]);
            }
        }
        result += term;
    }
    return result;
}

// Evaluate derivative of Lagrange polynomial at query point
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T evaluate_lagrange_polynomial_derivative(
    const T* x,      // node x-coordinates (n points)
    const T* y,      // node y-values (n points)
    T query,         // query point
    int64_t n
) {
    T result = T(0);
    for (int64_t j = 0; j < n; ++j) {
        result += y[j] * lagrange_basis_derivative(x, j, query, n);
    }
    return result;
}

}  // namespace torchscience::kernel::polynomial
```

#### Step 1.4: Create kernel header (backward)

Create `src/torchscience/csrc/kernel/polynomial/lagrange_polynomial_backward.h`:

```cpp
#pragma once

#include <cmath>

namespace torchscience::kernel::polynomial {

// Backward pass for Lagrange polynomial evaluation
// Computes gradients w.r.t. y values and query point
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void lagrange_polynomial_backward_single(
    T grad_output,
    const T* x,
    const T* y,
    T query,
    int64_t n,
    T* grad_y,            // gradient w.r.t. y (n elements)
    T* grad_query         // gradient w.r.t. query point
) {
    // Gradient w.r.t. y_j: grad_output * ℓ_j(query)
    for (int64_t j = 0; j < n; ++j) {
        grad_y[j] += grad_output * lagrange_basis(x, j, query, n);
    }

    // Gradient w.r.t. query: grad_output * L'(query)
    *grad_query = grad_output * evaluate_lagrange_polynomial_derivative(x, y, query, n);
}

}  // namespace torchscience::kernel::polynomial
```

#### Step 1.5: Create CPU implementation

Create `src/torchscience/csrc/cpu/polynomial/lagrange_polynomial.h`:

```cpp
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "kernel/polynomial/lagrange_polynomial.h"
#include "kernel/polynomial/lagrange_polynomial_backward.h"

namespace torchscience::cpu::polynomial {

inline at::Tensor lagrange_polynomial(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& query
) {
    TORCH_CHECK(x.dim() >= 1, "lagrange_polynomial: x must have at least 1 dimension");
    TORCH_CHECK(y.dim() >= 1, "lagrange_polynomial: y must have at least 1 dimension");
    TORCH_CHECK(x.size(-1) == y.size(-1), "lagrange_polynomial: x and y must have same last dimension");
    TORCH_CHECK(x.size(-1) >= 1, "lagrange_polynomial: need at least 1 point");

    auto x_contig = x.contiguous();
    auto y_contig = y.contiguous();
    auto query_contig = query.contiguous();

    int64_t n = x.size(-1);
    int64_t num_queries = query.numel();

    // Compute batch dimensions
    int64_t batch_size = 1;
    std::vector<int64_t> batch_dims;
    for (int64_t i = 0; i < x.dim() - 1; ++i) {
        batch_dims.push_back(x.size(i));
        batch_size *= x.size(i);
    }

    // Output shape: batch_dims + query_dims
    std::vector<int64_t> output_shape = batch_dims;
    for (int64_t i = 0; i < query.dim(); ++i) {
        output_shape.push_back(query.size(i));
    }

    auto output = at::empty(output_shape, y.options());

    AT_DISPATCH_FLOATING_TYPES(
        y.scalar_type(), "lagrange_polynomial_cpu", [&] {
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
            const scalar_t* query_ptr = query_contig.data_ptr<scalar_t>();
            scalar_t* output_ptr = output.data_ptr<scalar_t>();

            // Evaluate polynomial at query points
            int64_t queries_per_batch = num_queries / batch_size;
            at::parallel_for(0, num_queries, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    int64_t b = idx / queries_per_batch;
                    if (b >= batch_size) b = batch_size - 1;

                    scalar_t q = query_ptr[idx];
                    output_ptr[idx] = kernel::polynomial::evaluate_lagrange_polynomial(
                        x_ptr + b * n,
                        y_ptr + b * n,
                        q,
                        n
                    );
                }
            });
        }
    );

    return output;
}

inline std::tuple<at::Tensor, at::Tensor> lagrange_polynomial_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& query
) {
    auto x_contig = x.contiguous();
    auto y_contig = y.contiguous();
    auto query_contig = query.contiguous();
    auto grad_output_contig = grad_output.contiguous();

    int64_t n = x.size(-1);
    int64_t num_queries = query.numel();

    auto grad_y = at::zeros_like(y);
    auto grad_query = at::zeros_like(query);

    AT_DISPATCH_FLOATING_TYPES(
        y.scalar_type(), "lagrange_polynomial_backward_cpu", [&] {
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
            const scalar_t* query_ptr = query_contig.data_ptr<scalar_t>();
            const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
            scalar_t* grad_y_ptr = grad_y.data_ptr<scalar_t>();
            scalar_t* grad_query_ptr = grad_query.data_ptr<scalar_t>();

            // Process each query point (not parallelized due to atomic adds to grad_y)
            for (int64_t idx = 0; idx < num_queries; ++idx) {
                scalar_t q = query_ptr[idx];
                scalar_t g = grad_output_ptr[idx];

                // Gradient w.r.t. query
                grad_query_ptr[idx] = g * kernel::polynomial::evaluate_lagrange_polynomial_derivative(
                    x_ptr, y_ptr, q, n
                );

                // Gradient w.r.t. y
                for (int64_t j = 0; j < n; ++j) {
                    grad_y_ptr[j] += g * kernel::polynomial::lagrange_basis(x_ptr, j, q, n);
                }
            }
        }
    );

    return std::make_tuple(grad_y, grad_query);
}

}  // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("lagrange_polynomial", &torchscience::cpu::polynomial::lagrange_polynomial);
    m.impl("lagrange_polynomial_backward", &torchscience::cpu::polynomial::lagrange_polynomial_backward);
}
```

#### Step 1.6: Create Meta implementation

Create `src/torchscience/csrc/meta/polynomial/lagrange_polynomial.h`:

```cpp
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor lagrange_polynomial(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& query
) {
    TORCH_CHECK(x.dim() >= 1, "lagrange_polynomial: x must have at least 1 dimension");
    TORCH_CHECK(y.dim() >= 1, "lagrange_polynomial: y must have at least 1 dimension");
    TORCH_CHECK(x.size(-1) == y.size(-1), "lagrange_polynomial: x and y must have same last dimension");

    // Compute batch dimensions
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < x.dim() - 1; ++i) {
        output_shape.push_back(x.size(i));
    }
    for (int64_t i = 0; i < query.dim(); ++i) {
        output_shape.push_back(query.size(i));
    }

    return at::empty(output_shape, y.options());
}

inline std::tuple<at::Tensor, at::Tensor> lagrange_polynomial_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& query
) {
    return std::make_tuple(at::empty_like(y), at::empty_like(query));
}

}  // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("lagrange_polynomial", &torchscience::meta::polynomial::lagrange_polynomial);
    m.impl("lagrange_polynomial_backward", &torchscience::meta::polynomial::lagrange_polynomial_backward);
}
```

#### Step 1.7: Create Autograd wrapper

Create `src/torchscience/csrc/autograd/polynomial/lagrange_polynomial.h`:

```cpp
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::autograd::polynomial {

class LagrangePolynomialFunction : public torch::autograd::Function<LagrangePolynomialFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& x,
        const at::Tensor& y,
        const at::Tensor& query
    ) {
        ctx->save_for_backward({x, y, query});

        at::AutoDispatchBelowAutograd guard;
        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::lagrange_polynomial", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
        return op.call(x, y, query);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto y = saved[1];
        auto query = saved[2];
        auto grad_output = grad_outputs[0];

        at::AutoDispatchBelowAutograd guard;
        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::lagrange_polynomial_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>();

        auto [grad_y, grad_query] = op.call(grad_output, x, y, query);

        return {at::Tensor(), grad_y, grad_query};
    }
};

inline at::Tensor lagrange_polynomial(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& query
) {
    return LagrangePolynomialFunction::apply(x, y, query);
}

}  // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("lagrange_polynomial", &torchscience::autograd::polynomial::lagrange_polynomial);
}
```

#### Step 1.8: Add schema registration

Modify `src/torchscience/csrc/torchscience.cpp` to add:

```cpp
// In TORCH_LIBRARY(torchscience, m) block:
module.def("lagrange_polynomial(Tensor x, Tensor y, Tensor query) -> Tensor");
module.def("lagrange_polynomial_backward(Tensor grad_output, Tensor x, Tensor y, Tensor query) -> (Tensor, Tensor)");
```

Add includes:

```cpp
#include "cpu/polynomial/lagrange_polynomial.h"
#include "meta/polynomial/lagrange_polynomial.h"
#include "autograd/polynomial/lagrange_polynomial.h"
```

#### Step 1.9: Create Python module __init__.py

Create `src/torchscience/polynomial/__init__.py`:

```python
"""Polynomial interpolation functions."""

from torchscience.polynomial._lagrange_polynomial import lagrange_polynomial

__all__ = [
    "lagrange_polynomial",
]
```

#### Step 1.10: Create Python API wrapper

Create `src/torchscience/polynomial/_lagrange_polynomial.py`:

```python
"""Lagrange polynomial interpolation."""

import torch
from torch import Tensor

import torchscience._C


def lagrange_polynomial(
    x: Tensor,
    y: Tensor,
    query: Tensor,
) -> Tensor:
    """Evaluate Lagrange polynomial interpolation at query points.

    Given data points (x, y), computes the unique polynomial of degree n-1
    that passes through all n points and evaluates it at query points.

    The Lagrange polynomial is:
        L(t) = Σ_j y_j * ℓ_j(t)

    where ℓ_j(t) = Π_{k≠j} (t - x_k) / (x_j - x_k) are the Lagrange basis polynomials.

    Warning:
        For large n with uniformly spaced nodes, Lagrange interpolation can
        exhibit the Runge phenomenon (oscillation at edges). Consider using
        Chebyshev nodes or spline interpolation for better stability.

    Args:
        x: Node x-coordinates. Shape: (..., n) where n >= 1.
            Must have distinct values along the last dimension.
        y: Node y-values. Shape: (..., n), must match x.
        query: Points at which to evaluate the polynomial. Shape: (..., m).

    Returns:
        Interpolated values at query points. Shape: (..., m).

    Raises:
        ValueError: If x values contain duplicates.

    Example:
        >>> x = torch.tensor([0.0, 1.0, 2.0])
        >>> y = torch.tensor([1.0, 0.0, 1.0])  # Parabola
        >>> query = torch.tensor([0.5, 1.5])
        >>> result = lagrange_polynomial(x, y, query)
    """
    # Validate inputs
    if x.dim() < 1 or y.dim() < 1:
        raise ValueError("x and y must have at least 1 dimension")

    if x.size(-1) != y.size(-1):
        raise ValueError(f"x and y must have same last dimension, got {x.size(-1)} and {y.size(-1)}")

    if x.size(-1) < 1:
        raise ValueError(f"Need at least 1 point, got {x.size(-1)}")

    # Check distinct values (no duplicates)
    if x.size(-1) > 1:
        sorted_x = x.sort(dim=-1).values
        diffs = sorted_x[..., 1:] - sorted_x[..., :-1]
        if (diffs == 0).any():
            raise ValueError("x values must be distinct (no duplicates)")

    return torch.ops.torchscience.lagrange_polynomial(x, y, query)
```

#### Step 1.11: Run tests to verify

Run: `uv run pytest tests/torchscience/polynomial/test__lagrange_polynomial.py -v`
Expected: PASS

#### Step 1.12: Commit

```bash
git add tests/torchscience/polynomial/ src/torchscience/polynomial/ src/torchscience/csrc/kernel/polynomial/ src/torchscience/csrc/cpu/polynomial/ src/torchscience/csrc/meta/polynomial/ src/torchscience/csrc/autograd/polynomial/ src/torchscience/csrc/torchscience.cpp
git commit -m "feat(polynomial): add lagrange_polynomial operator"
```

---

### Task 2: Add `newton_polynomial` interpolation operator

**Goal:** Implement Newton polynomial interpolation using divided differences.

**Mathematical Definition:**
$$N(x) = \sum_{j=0}^{n} [y_0, \ldots, y_j] \prod_{k=0}^{j-1}(x - x_k)$$

where $[y_0, \ldots, y_j]$ are the divided differences:
- $[y_i] = y_i$
- $[y_i, y_{i+1}] = \frac{y_{i+1} - y_i}{x_{i+1} - x_i}$
- $[y_i, \ldots, y_{i+k}] = \frac{[y_{i+1}, \ldots, y_{i+k}] - [y_i, \ldots, y_{i+k-1}]}{x_{i+k} - x_i}$

**Files:**
- Create: `tests/torchscience/polynomial/test__newton_polynomial.py`
- Create: `src/torchscience/csrc/kernel/polynomial/newton_polynomial.h`
- Create: `src/torchscience/csrc/kernel/polynomial/newton_polynomial_backward.h`
- Create: `src/torchscience/csrc/cpu/polynomial/newton_polynomial.h`
- Create: `src/torchscience/csrc/meta/polynomial/newton_polynomial.h`
- Create: `src/torchscience/csrc/autograd/polynomial/newton_polynomial.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Modify: `src/torchscience/polynomial/__init__.py`
- Create: `src/torchscience/polynomial/_newton_polynomial.py`

Follow the same step pattern as Task 1.

**Commit:**
```bash
git commit -m "feat(polynomial): add newton_polynomial operator"
```

---

### Task 3: Add `barycentric_interpolation` operator

**Goal:** Implement numerically stable barycentric interpolation.

**Mathematical Definition:**
$$p(x) = \frac{\sum_{j=0}^{n} \frac{w_j}{x - x_j} y_j}{\sum_{j=0}^{n} \frac{w_j}{x - x_j}}$$

where $w_j = \prod_{k \neq j} \frac{1}{x_j - x_k}$ are the barycentric weights.

This formulation is O(n) for evaluation (after O(n²) preprocessing for weights) and numerically more stable than Lagrange or Newton forms.

**Files:**
- Create: `tests/torchscience/polynomial/test__barycentric_interpolation.py`
- Create: `src/torchscience/csrc/kernel/polynomial/barycentric_interpolation.h`
- Create: `src/torchscience/csrc/kernel/polynomial/barycentric_interpolation_backward.h`
- Create: `src/torchscience/csrc/cpu/polynomial/barycentric_interpolation.h`
- Create: `src/torchscience/csrc/meta/polynomial/barycentric_interpolation.h`
- Create: `src/torchscience/csrc/autograd/polynomial/barycentric_interpolation.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Modify: `src/torchscience/polynomial/__init__.py`
- Create: `src/torchscience/polynomial/_barycentric_interpolation.py`

Follow the same step pattern as Task 1.

**Commit:**
```bash
git commit -m "feat(polynomial): add barycentric_interpolation operator"
```

---

### Task 4: Add `chebyshev_nodes` operator

**Goal:** Generate Chebyshev nodes for optimal interpolation (minimizes Runge phenomenon).

**Mathematical Definition:**
First kind (Chebyshev-Gauss nodes):
$$x_k = \cos\left(\frac{2k-1}{2n}\pi\right), \quad k = 1, \ldots, n$$

Second kind (Chebyshev-Gauss-Lobatto nodes, includes endpoints):
$$x_k = \cos\left(\frac{k\pi}{n}\right), \quad k = 0, \ldots, n$$

Both give nodes in [-1, 1]. Can be scaled to [a, b] via linear transformation.

**Files:**
- Create: `tests/torchscience/polynomial/test__chebyshev_nodes.py`
- Create: `src/torchscience/csrc/kernel/polynomial/chebyshev_nodes.h`
- Create: `src/torchscience/csrc/cpu/polynomial/chebyshev_nodes.h`
- Create: `src/torchscience/csrc/meta/polynomial/chebyshev_nodes.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Modify: `src/torchscience/polynomial/__init__.py`
- Create: `src/torchscience/polynomial/_chebyshev_nodes.py`

Note: No backward needed for node generation (just indices).

Follow the same step pattern as Task 1.

**Commit:**
```bash
git commit -m "feat(polynomial): add chebyshev_nodes operator"
```

---

### Task 5: Add `chebyshev_interpolation` operator

**Goal:** Polynomial interpolation at Chebyshev nodes using Chebyshev polynomial basis.

**Mathematical Definition:**
$$p(x) = \sum_{k=0}^{n} c_k T_k(x)$$

where $T_k$ are Chebyshev polynomials of the first kind:
- $T_0(x) = 1$
- $T_1(x) = x$
- $T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)$

Coefficients are computed via discrete Chebyshev transform (DCT):
$$c_k = \frac{2}{n} \sum_{j=0}^{n-1} y_j T_k(x_j)$$

**Files:**
- Create: `tests/torchscience/polynomial/test__chebyshev_interpolation.py`
- Create: `src/torchscience/csrc/kernel/polynomial/chebyshev_interpolation.h`
- Create: `src/torchscience/csrc/kernel/polynomial/chebyshev_interpolation_backward.h`
- Create: `src/torchscience/csrc/cpu/polynomial/chebyshev_interpolation.h`
- Create: `src/torchscience/csrc/meta/polynomial/chebyshev_interpolation.h`
- Create: `src/torchscience/csrc/autograd/polynomial/chebyshev_interpolation.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Modify: `src/torchscience/polynomial/__init__.py`
- Create: `src/torchscience/polynomial/_chebyshev_interpolation.py`

Follow the same step pattern as Task 1.

**Commit:**
```bash
git commit -m "feat(polynomial): add chebyshev_interpolation operator"
```

---

## Implementation Notes

### C++ Kernel Pattern

All operators follow the established torchscience pattern:
1. Kernel template in `csrc/kernel/polynomial/`
2. CPU implementation in `csrc/cpu/polynomial/`
3. Meta implementation in `csrc/meta/polynomial/`
4. Autograd wrapper in `csrc/autograd/polynomial/`
5. Schema registration in `torchscience.cpp`
6. Python API in `src/torchscience/polynomial/`

### Autograd Considerations

For polynomial interpolation operators:
1. **Gradient w.r.t. y**: Direct since polynomial is linear in y values
2. **Gradient w.r.t. query**: Computed via polynomial derivative

### Batching Convention

All operators support batched inputs:
- Data: `(batch, n)` for n interpolation nodes
- Query: `(batch, m)`
- Output: `(batch, m)`

### Numerical Stability

- **Lagrange**: O(n²), prone to numerical instability for large n
- **Newton**: O(n²) setup, O(n) evaluation, better conditioning
- **Barycentric**: O(n²) setup, O(n) evaluation, most stable
- **Chebyshev**: Optimal for smooth functions, avoids Runge phenomenon

### Reference Libraries

Key references for implementation:
- [SciPy interpolate](https://docs.scipy.org/doc/scipy/reference/interpolate.html) - Comprehensive reference
- [Numerical Recipes](http://numerical.recipes/) - Chapter on interpolation
- [Chebfun](https://www.chebfun.org/) - Chebyshev approximation theory
- [DLMF Chapter 3](https://dlmf.nist.gov/3) - Mathematical foundations

---

## Status: READY FOR IMPLEMENTATION
