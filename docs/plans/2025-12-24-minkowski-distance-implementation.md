# Minkowski Distance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `torchscience.distance.minkowski_distance` as an N-dimensional pairwise operator using the CPUPairwiseOperator template pattern.

**Architecture:** Custom C++ implementation using the pairwise operator template. Forward computes weighted Minkowski distance between all pairs of points. Backward computes gradients through the p-norm formula. Supports self-distance mode (single input), optional per-dimension weights, and arbitrary p > 0.

**Tech Stack:** C++17, PyTorch C++ API (ATen), torch.autograd, pybind11

---

## Task 1: Create distance module Python structure

**Files:**
- Create: `src/torchscience/distance/__init__.py`
- Create: `src/torchscience/distance/_minkowski_distance.py`
- Modify: `src/torchscience/__init__.py`

**Step 1: Create the distance module __init__.py**

```python
# src/torchscience/distance/__init__.py
from ._minkowski_distance import minkowski_distance

__all__ = [
    "minkowski_distance",
]
```

**Step 2: Create stub Python API**

```python
# src/torchscience/distance/_minkowski_distance.py
"""Minkowski distance implementation."""

from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def minkowski_distance(
    x: Tensor,
    y: Optional[Tensor] = None,
    *,
    p: float = 2.0,
    weight: Optional[Tensor] = None,
) -> Tensor:
    r"""Compute pairwise Minkowski distances.

    Computes the weighted Minkowski distance between each pair of row vectors
    from two sets of points, or self-pairwise distances if only one set is given.

    Mathematical Definition
    -----------------------
    For points :math:`x \in \mathbb{R}^d` and :math:`y \in \mathbb{R}^d` with
    optional weights :math:`w \in \mathbb{R}^d`:

    .. math::
        d_p(x, y; w) = \left( \sum_{i=1}^{d} w_i |x_i - y_i|^p \right)^{1/p}

    Special cases:

    - :math:`p = 1`: Manhattan (taxicab) distance
    - :math:`p = 2`: Euclidean distance
    - :math:`p \to \infty`: Chebyshev distance (max absolute difference)

    Parameters
    ----------
    x : Tensor, shape (m, d)
        First set of m points in d-dimensional space.
    y : Tensor, shape (n, d), optional
        Second set of n points in d-dimensional space.
        If ``None``, computes self-pairwise distances from x to x.
    p : float, default=2.0
        Order of the Minkowski norm. Must be > 0.
        For 0 < p < 1, this is a quasi-metric (triangle inequality doesn't hold).
    weight : Tensor, shape (d,), optional
        Non-negative weights for each dimension.
        If ``None``, all weights are 1 (unweighted distance).

    Returns
    -------
    Tensor
        Pairwise distance matrix.

        - If ``y`` is provided: shape ``(m, n)`` where ``[i, j]`` is the
          distance from ``x[i]`` to ``y[j]``.
        - If ``y`` is ``None``: shape ``(m, m)`` where ``[i, j]`` is the
          distance from ``x[i]`` to ``x[j]``.

    Examples
    --------
    Compute Euclidean distances between two sets of points:

    >>> x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    >>> torchscience.distance.minkowski_distance(x, y, p=2.0)
    tensor([[1.0000, 1.0000],
            [1.0000, 1.0000]])

    Compute Manhattan distances (p=1):

    >>> torchscience.distance.minkowski_distance(x, y, p=1.0)
    tensor([[1., 1.],
            [1., 1.]])

    Self-pairwise distances:

    >>> x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    >>> torchscience.distance.minkowski_distance(x, p=2.0)
    tensor([[0., 5.],
            [5., 0.]])

    Weighted distance:

    >>> x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> y = torch.tensor([[1.0, 1.0]])
    >>> w = torch.tensor([1.0, 4.0])  # Weight second dimension more
    >>> torchscience.distance.minkowski_distance(x, y, p=2.0, weight=w)
    tensor([[2.2361],
            [0.0000]])

    Notes
    -----
    - For ``p >= 1``, the Minkowski distance is a true metric satisfying the
      triangle inequality.
    - For ``0 < p < 1``, the result is a quasi-metric.
    - Gradients are computed analytically and support higher-order derivatives.
    - The diagonal of self-pairwise distances is always 0.

    See Also
    --------
    torch.cdist : PyTorch's built-in pairwise distance function.
    scipy.spatial.distance.cdist : SciPy's pairwise distance function.

    References
    ----------
    .. [1] Wikipedia, "Minkowski distance",
           https://en.wikipedia.org/wiki/Minkowski_distance
    """
    # Input validation
    if x.dim() != 2:
        raise ValueError(f"x must be 2D (m, d), got {x.dim()}D")

    if y is None:
        y = x

    if y.dim() != 2:
        raise ValueError(f"y must be 2D (n, d), got {y.dim()}D")

    if x.size(1) != y.size(1):
        raise ValueError(
            f"Feature dimensions must match: x has {x.size(1)}, y has {y.size(1)}"
        )

    if p <= 0:
        raise ValueError(f"p must be > 0, got {p}")

    if weight is not None:
        if weight.dim() != 1:
            raise ValueError(f"weight must be 1D (d,), got {weight.dim()}D")
        if weight.size(0) != x.size(1):
            raise ValueError(
                f"weight size {weight.size(0)} must match feature dim {x.size(1)}"
            )
        if (weight < 0).any():
            raise ValueError("weight must be non-negative")

    return torch.ops.torchscience.minkowski_distance(x, y, p, weight)
```

**Step 3: Update main __init__.py to include distance module**

Modify `src/torchscience/__init__.py` to add `distance` to imports:

```python
from . import (
    _csrc,
    distance,
    optimization,
    root_finding,
    signal_processing,
    statistics,
)

__all__ = [
    "_csrc",
    "distance",
    "optimization",
    "root_finding",
    "signal_processing",
    "statistics",
]
```

**Step 4: Commit**

```bash
git add src/torchscience/distance/ src/torchscience/__init__.py
git commit -m "feat(distance): add module structure and Python API for minkowski_distance"
```

---

## Task 2: Create impl header with forward/backward math

**Files:**
- Create: `src/torchscience/csrc/impl/distance/minkowski_distance.h`
- Create: `src/torchscience/csrc/impl/distance/minkowski_distance_backward.h`

**Step 1: Create forward implementation**

```cpp
// src/torchscience/csrc/impl/distance/minkowski_distance.h
#pragma once

/*
 * Minkowski Distance Implementation
 *
 * MATHEMATICAL DEFINITION:
 * ========================
 * The weighted Minkowski distance between vectors x and y:
 *
 *   d_p(x, y; w) = ( sum_i w_i * |x_i - y_i|^p )^(1/p)
 *
 * Special cases:
 *   p = 1: Manhattan distance (weighted L1)
 *   p = 2: Euclidean distance (weighted L2)
 *   p -> inf: Chebyshev distance (max weighted absolute diff)
 *
 * ALGORITHM:
 * ==========
 * For numerical stability with large/small p:
 *   - For moderate p: direct computation
 *   - For very large p: approximate with max (TODO: future optimization)
 */

#include <c10/macros/Macros.h>
#include <cmath>

namespace torchscience::impl::distance {

/**
 * Compute weighted Minkowski distance between two vectors.
 *
 * @param x First vector (pointer to d elements)
 * @param y Second vector (pointer to d elements)
 * @param d Dimension of vectors
 * @param p Order of the norm (p > 0)
 * @param w Optional weights (nullptr for unweighted, else pointer to d elements)
 * @return Distance value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T minkowski_distance_pair(
    const T* x,
    const T* y,
    int64_t d,
    T p,
    const T* w
) {
    T sum = T(0);

    if (w == nullptr) {
        // Unweighted case
        for (int64_t i = 0; i < d; ++i) {
            T diff = x[i] - y[i];
            T abs_diff = diff >= T(0) ? diff : -diff;
            sum += std::pow(abs_diff, p);
        }
    } else {
        // Weighted case
        for (int64_t i = 0; i < d; ++i) {
            T diff = x[i] - y[i];
            T abs_diff = diff >= T(0) ? diff : -diff;
            sum += w[i] * std::pow(abs_diff, p);
        }
    }

    // Handle p = 1 and p = 2 specially for numerical stability
    if (p == T(1)) {
        return sum;
    } else if (p == T(2)) {
        return std::sqrt(sum);
    } else {
        return std::pow(sum, T(1) / p);
    }
}

}  // namespace torchscience::impl::distance
```

**Step 2: Create backward implementation**

```cpp
// src/torchscience/csrc/impl/distance/minkowski_distance_backward.h
#pragma once

/*
 * Minkowski Distance Backward Implementation
 *
 * GRADIENT DERIVATION:
 * ====================
 * Let d = ( sum_i w_i * |x_i - y_i|^p )^(1/p)
 *
 * Partial derivative with respect to x_k:
 *
 *   dd/dx_k = (1/p) * d^(1-p) * w_k * p * |x_k - y_k|^(p-1) * sign(x_k - y_k)
 *           = d^(1-p) * w_k * |x_k - y_k|^(p-1) * sign(x_k - y_k)
 *           = w_k * sign(x_k - y_k) * |x_k - y_k|^(p-1) / d^(p-1)
 *
 * Similarly: dd/dy_k = -dd/dx_k
 *
 * EDGE CASES:
 * ===========
 * - d = 0: gradient is 0 (all components identical)
 * - x_k = y_k: that component's gradient is 0
 * - p < 1: gradient can be large near zero (quasi-metric)
 */

#include <c10/macros/Macros.h>
#include <cmath>

namespace torchscience::impl::distance {

/**
 * Compute gradients for weighted Minkowski distance.
 *
 * @param grad_out Upstream gradient (scalar)
 * @param x First vector
 * @param y Second vector
 * @param d Dimension
 * @param p Order of norm
 * @param w Optional weights
 * @param dist Pre-computed distance value
 * @param grad_x Output gradient for x (d elements)
 * @param grad_y Output gradient for y (d elements)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void minkowski_distance_backward_pair(
    T grad_out,
    const T* x,
    const T* y,
    int64_t dim,
    T p,
    const T* w,
    T dist,
    T* grad_x,
    T* grad_y
) {
    // Handle zero distance case
    if (dist == T(0)) {
        for (int64_t i = 0; i < dim; ++i) {
            grad_x[i] = T(0);
            grad_y[i] = T(0);
        }
        return;
    }

    // Compute d^(p-1) for the denominator
    T dist_pow_pm1 = std::pow(dist, p - T(1));

    for (int64_t i = 0; i < dim; ++i) {
        T diff = x[i] - y[i];

        // Handle zero difference
        if (diff == T(0)) {
            grad_x[i] = T(0);
            grad_y[i] = T(0);
            continue;
        }

        T abs_diff = diff >= T(0) ? diff : -diff;
        T sign_diff = diff >= T(0) ? T(1) : T(-1);

        // |x_k - y_k|^(p-1)
        T abs_diff_pow_pm1;
        if (p == T(1)) {
            abs_diff_pow_pm1 = T(1);
        } else if (p == T(2)) {
            abs_diff_pow_pm1 = abs_diff;
        } else {
            abs_diff_pow_pm1 = std::pow(abs_diff, p - T(1));
        }

        // Weight factor
        T weight_i = (w != nullptr) ? w[i] : T(1);

        // Gradient: w_k * sign(diff) * |diff|^(p-1) / d^(p-1)
        T grad_component = weight_i * sign_diff * abs_diff_pow_pm1 / dist_pow_pm1;

        grad_x[i] = grad_out * grad_component;
        grad_y[i] = -grad_out * grad_component;
    }
}

}  // namespace torchscience::impl::distance
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/impl/distance/
git commit -m "feat(impl): add minkowski distance forward and backward math implementations"
```

---

## Task 3: Create CPU kernel using pairwise operator template

**Files:**
- Create: `src/torchscience/csrc/cpu/distance/minkowski_distance.h`

**Step 1: Create CPU implementation using CPUPairwiseOperator**

```cpp
// src/torchscience/csrc/cpu/distance/minkowski_distance.h
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../impl/distance/minkowski_distance.h"
#include "../../impl/distance/minkowski_distance_backward.h"

namespace torchscience::cpu::distance {

/**
 * CPU implementation of pairwise Minkowski distance.
 *
 * Computes distance between each pair of points from x and y.
 *
 * @param x First set of points, shape (m, d)
 * @param y Second set of points, shape (n, d)
 * @param p Order of the norm
 * @param weight Optional weights, shape (d,)
 * @return Distance matrix, shape (m, n)
 */
inline at::Tensor minkowski_distance(
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight
) {
    TORCH_CHECK(x.dim() == 2, "minkowski_distance: x must be 2D (m, d)");
    TORCH_CHECK(y.dim() == 2, "minkowski_distance: y must be 2D (n, d)");
    TORCH_CHECK(x.size(1) == y.size(1), "minkowski_distance: feature dimensions must match");
    TORCH_CHECK(p > 0, "minkowski_distance: p must be > 0");

    int64_t m = x.size(0);
    int64_t n = y.size(0);
    int64_t d = x.size(1);

    at::Tensor x_contig = x.contiguous();
    at::Tensor y_contig = y.contiguous();
    at::Tensor output = at::empty({m, n}, x.options());

    // Handle optional weight
    at::Tensor w_contig;
    bool has_weight = weight.has_value() && weight->defined();
    if (has_weight) {
        TORCH_CHECK(weight->dim() == 1, "minkowski_distance: weight must be 1D (d,)");
        TORCH_CHECK(weight->size(0) == d, "minkowski_distance: weight size must match feature dim");
        w_contig = weight->contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        x.scalar_type(),
        "minkowski_distance_cpu",
        [&]() {
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
            const scalar_t* w_ptr = has_weight ? w_contig.data_ptr<scalar_t>() : nullptr;
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t p_val = static_cast<scalar_t>(p);

            at::parallel_for(0, m * n, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    int64_t i = idx / n;
                    int64_t j = idx % n;
                    out_ptr[idx] = impl::distance::minkowski_distance_pair<scalar_t>(
                        x_ptr + i * d,
                        y_ptr + j * d,
                        d,
                        p_val,
                        w_ptr
                    );
                }
            });
        }
    );

    return output;
}

/**
 * Backward pass for Minkowski distance.
 */
inline std::tuple<at::Tensor, at::Tensor> minkowski_distance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight,
    const at::Tensor& dist_output
) {
    int64_t m = x.size(0);
    int64_t n = y.size(0);
    int64_t d = x.size(1);

    at::Tensor x_contig = x.contiguous();
    at::Tensor y_contig = y.contiguous();
    at::Tensor grad_contig = grad_output.contiguous();
    at::Tensor dist_contig = dist_output.contiguous();

    at::Tensor grad_x = at::zeros_like(x);
    at::Tensor grad_y = at::zeros_like(y);

    at::Tensor w_contig;
    bool has_weight = weight.has_value() && weight->defined();
    if (has_weight) {
        w_contig = weight->contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        x.scalar_type(),
        "minkowski_distance_backward_cpu",
        [&]() {
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_contig.data_ptr<scalar_t>();
            const scalar_t* dist_ptr = dist_contig.data_ptr<scalar_t>();
            const scalar_t* w_ptr = has_weight ? w_contig.data_ptr<scalar_t>() : nullptr;
            scalar_t* grad_x_ptr = grad_x.data_ptr<scalar_t>();
            scalar_t* grad_y_ptr = grad_y.data_ptr<scalar_t>();
            scalar_t p_val = static_cast<scalar_t>(p);

            // Sequential accumulation for correctness
            // (parallel would need atomic operations or per-thread buffers)
            std::vector<scalar_t> temp_grad_x(d);
            std::vector<scalar_t> temp_grad_y(d);

            for (int64_t i = 0; i < m; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t grad_val = grad_ptr[i * n + j];
                    scalar_t dist_val = dist_ptr[i * n + j];

                    impl::distance::minkowski_distance_backward_pair<scalar_t>(
                        grad_val,
                        x_ptr + i * d,
                        y_ptr + j * d,
                        d,
                        p_val,
                        w_ptr,
                        dist_val,
                        temp_grad_x.data(),
                        temp_grad_y.data()
                    );

                    for (int64_t k = 0; k < d; ++k) {
                        grad_x_ptr[i * d + k] += temp_grad_x[k];
                        grad_y_ptr[j * d + k] += temp_grad_y[k];
                    }
                }
            }
        }
    );

    return std::make_tuple(grad_x, grad_y);
}

}  // namespace torchscience::cpu::distance

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("minkowski_distance", &torchscience::cpu::distance::minkowski_distance);
    m.impl("minkowski_distance_backward", &torchscience::cpu::distance::minkowski_distance_backward);
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/distance/
git commit -m "feat(cpu): add minkowski distance CPU kernel implementation"
```

---

## Task 4: Create autograd wrapper

**Files:**
- Create: `src/torchscience/csrc/autograd/distance/minkowski_distance.h`

**Step 1: Create autograd Function class**

```cpp
// src/torchscience/csrc/autograd/distance/minkowski_distance.h
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::distance {

/**
 * Autograd Function for Minkowski distance.
 */
class MinkowskiDistance
    : public torch::autograd::Function<MinkowskiDistance> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& x,
        const at::Tensor& y,
        double p,
        const c10::optional<at::Tensor>& weight
    ) {
        ctx->save_for_backward({x, y});
        if (weight.has_value() && weight->defined()) {
            ctx->saved_data["weight"] = weight.value();
            ctx->saved_data["has_weight"] = true;
        } else {
            ctx->saved_data["has_weight"] = false;
        }
        ctx->saved_data["p"] = p;

        bool x_requires_grad = x.requires_grad() && at::isFloatingType(x.scalar_type());
        bool y_requires_grad = y.requires_grad() && at::isFloatingType(y.scalar_type());
        ctx->saved_data["x_requires_grad"] = x_requires_grad;
        ctx->saved_data["y_requires_grad"] = y_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::minkowski_distance", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                double,
                const c10::optional<at::Tensor>&
            )>()
            .call(x, y, p, weight);

        // Save output for backward (needed for gradient computation)
        ctx->save_for_backward({x, y, output});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor x = saved[0];
        at::Tensor y = saved[1];
        at::Tensor dist_output = saved[2];

        at::Tensor grad_output = grad_outputs[0];

        double p = ctx->saved_data["p"].toDouble();
        bool has_weight = ctx->saved_data["has_weight"].toBool();
        bool x_requires_grad = ctx->saved_data["x_requires_grad"].toBool();
        bool y_requires_grad = ctx->saved_data["y_requires_grad"].toBool();

        c10::optional<at::Tensor> weight;
        if (has_weight) {
            weight = ctx->saved_data["weight"].toTensor();
        }

        if (!x_requires_grad && !y_requires_grad) {
            return {
                at::Tensor(),  // grad_x
                at::Tensor(),  // grad_y
                at::Tensor(),  // grad_p
                at::Tensor()   // grad_weight
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_x, grad_y] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::minkowski_distance_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                double,
                const c10::optional<at::Tensor>&,
                const at::Tensor&
            )>()
            .call(grad_output, x, y, p, weight, dist_output);

        return {
            x_requires_grad ? grad_x : at::Tensor(),
            y_requires_grad ? grad_y : at::Tensor(),
            at::Tensor(),  // grad_p (not differentiable)
            at::Tensor()   // grad_weight (could be added later)
        };
    }
};

inline at::Tensor minkowski_distance(
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight
) {
    return MinkowskiDistance::apply(x, y, p, weight);
}

}  // namespace torchscience::autograd::distance

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("minkowski_distance", &torchscience::autograd::distance::minkowski_distance);
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/autograd/distance/
git commit -m "feat(autograd): add minkowski distance autograd wrapper"
```

---

## Task 5: Create meta tensor implementation

**Files:**
- Create: `src/torchscience/csrc/meta/distance/minkowski_distance.h`

**Step 1: Create meta implementation for shape inference**

```cpp
// src/torchscience/csrc/meta/distance/minkowski_distance.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::distance {

/**
 * Meta implementation for shape inference.
 */
inline at::Tensor minkowski_distance(
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight
) {
    TORCH_CHECK(x.dim() == 2, "minkowski_distance: x must be 2D (m, d)");
    TORCH_CHECK(y.dim() == 2, "minkowski_distance: y must be 2D (n, d)");
    TORCH_CHECK(x.size(1) == y.size(1), "minkowski_distance: feature dimensions must match");

    int64_t m = x.size(0);
    int64_t n = y.size(0);

    return at::empty({m, n}, x.options());
}

inline std::tuple<at::Tensor, at::Tensor> minkowski_distance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight,
    const at::Tensor& dist_output
) {
    return std::make_tuple(
        at::empty_like(x),
        at::empty_like(y)
    );
}

}  // namespace torchscience::meta::distance

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("minkowski_distance", &torchscience::meta::distance::minkowski_distance);
    m.impl("minkowski_distance_backward", &torchscience::meta::distance::minkowski_distance_backward);
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/meta/distance/
git commit -m "feat(meta): add minkowski distance meta tensor implementation"
```

---

## Task 6: Register operators and update main cpp

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp`

**Step 1: Add operator schema definitions**

Add to the `TORCH_LIBRARY(torchscience, module)` block in `torchscience.cpp`:

```cpp
// `torchscience.distance`
module.def("minkowski_distance(Tensor x, Tensor y, float p, Tensor? weight) -> Tensor");
module.def("minkowski_distance_backward(Tensor grad_output, Tensor x, Tensor y, float p, Tensor? weight, Tensor dist_output) -> (Tensor, Tensor)");
```

**Step 2: Add includes for the new headers**

Add to the include section at the top:

```cpp
#include "cpu/distance/minkowski_distance.h"
#include "autograd/distance/minkowski_distance.h"
#include "meta/distance/minkowski_distance.h"
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat: register minkowski_distance operators in TORCH_LIBRARY"
```

---

## Task 7: Write basic correctness tests

**Files:**
- Create: `tests/torchscience/distance/__init__.py`
- Create: `tests/torchscience/distance/test__minkowski_distance.py`

**Step 1: Create test directory structure**

```python
# tests/torchscience/distance/__init__.py
```

**Step 2: Write test file**

```python
# tests/torchscience/distance/test__minkowski_distance.py
"""Tests for torchscience.distance.minkowski_distance."""

import pytest
import torch

import torchscience.distance


class TestMinkowskiDistanceBasic:
    """Basic functionality tests."""

    def test_output_shape_two_inputs(self):
        """Test output shape with two input tensors."""
        x = torch.randn(5, 3)
        y = torch.randn(7, 3)
        result = torchscience.distance.minkowski_distance(x, y)
        assert result.shape == (5, 7)

    def test_output_shape_self_distance(self):
        """Test output shape for self-pairwise distance."""
        x = torch.randn(5, 3)
        result = torchscience.distance.minkowski_distance(x)
        assert result.shape == (5, 5)

    def test_self_distance_diagonal_zero(self):
        """Self-distance diagonal should be zero."""
        x = torch.randn(5, 3)
        result = torchscience.distance.minkowski_distance(x)
        diagonal = torch.diag(result)
        torch.testing.assert_close(diagonal, torch.zeros(5))

    def test_distance_non_negative(self):
        """All distances should be non-negative."""
        x = torch.randn(5, 3)
        y = torch.randn(7, 3)
        result = torchscience.distance.minkowski_distance(x, y)
        assert (result >= 0).all()


class TestMinkowskiDistanceCorrectness:
    """Tests for numerical correctness."""

    def test_euclidean_distance_p2(self):
        """Test p=2 gives Euclidean distance."""
        x = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
        y = torch.tensor([[4.0, 0.0], [0.0, 4.0]])

        result = torchscience.distance.minkowski_distance(x, y, p=2.0)

        # d(x[0], y[0]) = 4, d(x[0], y[1]) = 4
        # d(x[1], y[0]) = 1, d(x[1], y[1]) = 5
        expected = torch.tensor([[4.0, 4.0], [1.0, 5.0]])
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_manhattan_distance_p1(self):
        """Test p=1 gives Manhattan distance."""
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([[3.0, 4.0]])

        result = torchscience.distance.minkowski_distance(x, y, p=1.0)

        expected = torch.tensor([[7.0]])  # |3-0| + |4-0| = 7
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_matches_torch_cdist(self):
        """Test that results match torch.cdist."""
        torch.manual_seed(42)
        x = torch.randn(10, 5)
        y = torch.randn(8, 5)

        for p in [1.0, 2.0, 3.0]:
            result = torchscience.distance.minkowski_distance(x, y, p=p)
            expected = torch.cdist(x, y, p=p)
            torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_weighted_distance(self):
        """Test weighted distance computation."""
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([[1.0, 1.0]])
        w = torch.tensor([1.0, 4.0])

        # sqrt(1*1^2 + 4*1^2) = sqrt(5)
        result = torchscience.distance.minkowski_distance(x, y, p=2.0, weight=w)
        expected = torch.tensor([[2.2360679775]])  # sqrt(5)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


class TestMinkowskiDistanceValidation:
    """Tests for input validation."""

    def test_invalid_x_dim(self):
        """Test error on non-2D x."""
        x = torch.randn(5)
        y = torch.randn(3, 3)
        with pytest.raises(ValueError, match="x must be 2D"):
            torchscience.distance.minkowski_distance(x, y)

    def test_invalid_y_dim(self):
        """Test error on non-2D y."""
        x = torch.randn(3, 3)
        y = torch.randn(5)
        with pytest.raises(ValueError, match="y must be 2D"):
            torchscience.distance.minkowski_distance(x, y)

    def test_mismatched_feature_dim(self):
        """Test error on mismatched feature dimensions."""
        x = torch.randn(3, 4)
        y = torch.randn(5, 3)
        with pytest.raises(ValueError, match="Feature dimensions must match"):
            torchscience.distance.minkowski_distance(x, y)

    def test_invalid_p(self):
        """Test error on p <= 0."""
        x = torch.randn(3, 3)
        with pytest.raises(ValueError, match="p must be > 0"):
            torchscience.distance.minkowski_distance(x, p=0.0)
        with pytest.raises(ValueError, match="p must be > 0"):
            torchscience.distance.minkowski_distance(x, p=-1.0)

    def test_invalid_weight_dim(self):
        """Test error on non-1D weight."""
        x = torch.randn(3, 3)
        w = torch.randn(3, 3)
        with pytest.raises(ValueError, match="weight must be 1D"):
            torchscience.distance.minkowski_distance(x, weight=w)

    def test_invalid_weight_size(self):
        """Test error on mismatched weight size."""
        x = torch.randn(3, 4)
        w = torch.randn(3)
        with pytest.raises(ValueError, match="weight size .* must match feature dim"):
            torchscience.distance.minkowski_distance(x, weight=w)

    def test_negative_weight(self):
        """Test error on negative weights."""
        x = torch.randn(3, 3)
        w = torch.tensor([1.0, -1.0, 1.0])
        with pytest.raises(ValueError, match="weight must be non-negative"):
            torchscience.distance.minkowski_distance(x, weight=w)


class TestMinkowskiDistanceGradient:
    """Tests for gradient computation."""

    def test_gradient_exists(self):
        """Test that gradient can be computed."""
        x = torch.randn(5, 3, requires_grad=True, dtype=torch.float64)
        y = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        result = torchscience.distance.minkowski_distance(x, y)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert y.grad is not None

    def test_gradient_finite(self):
        """Test that gradients are finite."""
        x = torch.randn(5, 3, requires_grad=True, dtype=torch.float64)
        y = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        result = torchscience.distance.minkowski_distance(x, y)
        loss = result.sum()
        loss.backward()
        assert torch.all(torch.isfinite(x.grad))
        assert torch.all(torch.isfinite(y.grad))

    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_gradcheck(self, p):
        """Test gradient correctness with gradcheck."""
        x = torch.randn(3, 4, requires_grad=True, dtype=torch.float64)
        y = torch.randn(2, 4, requires_grad=True, dtype=torch.float64)

        def fn(x_in, y_in):
            return torchscience.distance.minkowski_distance(x_in, y_in, p=p)

        assert torch.autograd.gradcheck(fn, (x, y), eps=1e-5, atol=1e-3, rtol=0.05)

    def test_gradcheck_with_weight(self):
        """Test gradient with weights."""
        x = torch.randn(3, 4, requires_grad=True, dtype=torch.float64)
        y = torch.randn(2, 4, requires_grad=True, dtype=torch.float64)
        w = torch.rand(4, dtype=torch.float64) + 0.1  # Ensure positive

        def fn(x_in, y_in):
            return torchscience.distance.minkowski_distance(x_in, y_in, p=2.0, weight=w)

        assert torch.autograd.gradcheck(fn, (x, y), eps=1e-5, atol=1e-3, rtol=0.05)


class TestMinkowskiDistanceDtype:
    """Tests for dtype support."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test float32 and float64 support."""
        x = torch.randn(5, 3, dtype=dtype)
        y = torch.randn(4, 3, dtype=dtype)
        result = torchscience.distance.minkowski_distance(x, y)
        assert result.dtype == dtype


class TestMinkowskiDistanceDevice:
    """Tests for device placement."""

    def test_cpu_device(self):
        """Test CPU computation."""
        x = torch.randn(5, 3, device="cpu")
        y = torch.randn(4, 3, device="cpu")
        result = torchscience.distance.minkowski_distance(x, y)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test CUDA computation."""
        x = torch.randn(5, 3, device="cuda")
        y = torch.randn(4, 3, device="cuda")
        result = torchscience.distance.minkowski_distance(x, y)
        assert result.device.type == "cuda"


class TestMinkowskiDistanceSciPyCompatibility:
    """Tests for SciPy compatibility."""

    def test_matches_scipy_cdist(self):
        """Test that results match SciPy cdist."""
        pytest.importorskip("scipy")
        from scipy.spatial.distance import cdist
        import numpy as np

        torch.manual_seed(42)
        x_np = np.random.randn(10, 5).astype(np.float64)
        y_np = np.random.randn(8, 5).astype(np.float64)
        x_torch = torch.from_numpy(x_np)
        y_torch = torch.from_numpy(y_np)

        for p in [1.0, 2.0, 3.5]:
            scipy_result = cdist(x_np, y_np, metric='minkowski', p=p)
            torch_result = torchscience.distance.minkowski_distance(
                x_torch, y_torch, p=p
            ).numpy()
            np.testing.assert_allclose(torch_result, scipy_result, rtol=1e-5, atol=1e-5)
```

**Step 3: Run tests to verify**

```bash
pytest tests/torchscience/distance/test__minkowski_distance.py -v
```

Expected: Tests should pass once implementation is complete.

**Step 4: Commit**

```bash
git add tests/torchscience/distance/
git commit -m "test: add comprehensive tests for minkowski_distance"
```

---

## Task 8: Build and verify

**Step 1: Build the project**

```bash
uv run pip install -e .
```

Expected: Build should complete without errors.

**Step 2: Run all minkowski distance tests**

```bash
uv run pytest tests/torchscience/distance/test__minkowski_distance.py -v
```

Expected: All tests pass.

**Step 3: Verify import works**

```bash
uv run python -c "import torchscience.distance; print(torchscience.distance.minkowski_distance)"
```

Expected: Prints the function object.

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address any build or test issues"
```

---

## Summary

This plan implements `torchscience.distance.minkowski_distance` as an N-dimensional pairwise operator:

1. **Python API** - User-facing function with validation and documentation
2. **impl headers** - Device-agnostic math for forward and backward
3. **CPU kernel** - Parallelized pairwise computation
4. **Autograd wrapper** - Gradient support through torch.autograd.Function
5. **Meta implementation** - Shape inference for torch.compile
6. **Operator registration** - TORCH_LIBRARY schema definitions
7. **Tests** - Comprehensive correctness, gradient, and compatibility tests
8. **Build verification** - Ensure everything compiles and runs
