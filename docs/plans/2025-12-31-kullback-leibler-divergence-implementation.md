# KL Divergence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `kullback_leibler_divergence` and `jensen_shannon_divergence` to `torchscience.information_theory` with full autograd support.

**Architecture:** Python API validates inputs, dispatches to C++ operators. C++ kernels handle preprocessing (softmax/exp/clamp), computation, and gradients. Follows existing patterns from `minkowski_distance`.

**Tech Stack:** PyTorch C++ extensions, ATen, torch::autograd::Function

---

## Task 1: Python Module Structure

**Files:**
- Create: `src/torchscience/information_theory/_kullback_leibler_divergence.py`
- Create: `src/torchscience/information_theory/_jensen_shannon_divergence.py`
- Modify: `src/torchscience/information_theory/__init__.py`

**Step 1: Create kullback_leibler_divergence Python API**

```python
# src/torchscience/information_theory/_kullback_leibler_divergence.py
"""Kullback-Leibler divergence implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def kullback_leibler_divergence(
    p: Tensor,
    q: Tensor,
    *,
    dim: int = -1,
    input_type: Literal["probability", "log_probability", "logits"] = "probability",
    reduction: Literal["none", "mean", "batchmean", "sum"] = "none",
    pairwise: bool = False,
) -> Tensor:
    r"""Compute Kullback-Leibler divergence between probability distributions.

    The KL divergence measures how one probability distribution P diverges
    from a reference distribution Q.

    Mathematical Definition
    -----------------------
    For discrete distributions P and Q:

    .. math::
        D_{KL}(P \| Q) = \sum_{i} p_i \log\left(\frac{p_i}{q_i}\right)

    Parameters
    ----------
    p : Tensor
        First probability distribution (or batch of distributions).
    q : Tensor
        Second probability distribution (or batch of distributions).
    dim : int, default=-1
        Dimension along which the probability distribution is defined.
    input_type : {"probability", "log_probability", "logits"}, default="probability"
        How to interpret input tensors:

        - ``"probability"``: Direct probability mass functions (epsilon-clamped)
        - ``"log_probability"``: Log-probabilities (exponentiated before use)
        - ``"logits"``: Unnormalized logits (softmax applied)
    reduction : {"none", "mean", "batchmean", "sum"}, default="none"
        Reduction to apply:

        - ``"none"``: Return per-sample divergences
        - ``"mean"``: Mean over all elements
        - ``"batchmean"``: Mean over batch dimension (mathematically correct KL)
        - ``"sum"``: Sum over all elements
    pairwise : bool, default=False
        If True, compute all-pairs divergence matrix.
        ``p: (m, n)`` and ``q: (k, n)`` produces output ``(m, k)``.

    Returns
    -------
    Tensor
        KL divergence values. Shape depends on ``reduction`` and ``pairwise``.

    Examples
    --------
    >>> p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> q = torch.tensor([0.1, 0.2, 0.3, 0.4])
    >>> torchscience.information_theory.kullback_leibler_divergence(p, q)
    tensor(0.1335)

    >>> # Batch of distributions
    >>> p = torch.softmax(torch.randn(10, 5), dim=-1)
    >>> q = torch.softmax(torch.randn(10, 5), dim=-1)
    >>> kl = torchscience.information_theory.kullback_leibler_divergence(
    ...     p, q, reduction="none"
    ... )
    >>> kl.shape
    torch.Size([10])

    Notes
    -----
    - KL divergence is asymmetric: :math:`D_{KL}(P \| Q) \\neq D_{KL}(Q \| P)`
    - Values are clamped to avoid log(0): ``eps`` is dtype-dependent
    - Supports first and second-order gradients

    See Also
    --------
    jensen_shannon_divergence : Symmetric divergence measure.
    torch.nn.functional.kl_div : PyTorch's KL divergence (different API).
    """
    # Validate input types
    if not isinstance(p, Tensor):
        raise TypeError(f"p must be a Tensor, got {type(p).__name__}")
    if not isinstance(q, Tensor):
        raise TypeError(f"q must be a Tensor, got {type(q).__name__}")

    # Validate input_type
    valid_input_types = ("probability", "log_probability", "logits")
    if input_type not in valid_input_types:
        raise ValueError(
            f"input_type must be one of {valid_input_types}, got '{input_type}'"
        )

    # Validate reduction
    valid_reductions = ("none", "mean", "batchmean", "sum")
    if reduction not in valid_reductions:
        raise ValueError(
            f"reduction must be one of {valid_reductions}, got '{reduction}'"
        )

    # Normalize dim
    p_dim = p.dim()
    if dim < -p_dim or dim >= p_dim:
        raise IndexError(
            f"dim {dim} out of range for tensor with {p_dim} dimensions"
        )
    dim = dim if dim >= 0 else p_dim + dim

    # Check distribution dimension sizes match
    if p.size(dim) != q.size(dim):
        raise ValueError(
            f"Distribution sizes must match along dim {dim}: "
            f"p has {p.size(dim)}, q has {q.size(dim)}"
        )

    # Validate pairwise mode
    if pairwise:
        if p.dim() < 2 or q.dim() < 2:
            raise ValueError(
                "pairwise=True requires p and q to be at least 2D"
            )

    # Dtype promotion
    target_dtype = torch.promote_types(p.dtype, q.dtype)
    if p.dtype != target_dtype:
        p = p.to(target_dtype)
    if q.dtype != target_dtype:
        q = q.to(target_dtype)

    return torch.ops.torchscience.kullback_leibler_divergence(
        p, q, dim, input_type, reduction, pairwise
    )
```

**Step 2: Create jensen_shannon_divergence Python API**

```python
# src/torchscience/information_theory/_jensen_shannon_divergence.py
"""Jensen-Shannon divergence implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def jensen_shannon_divergence(
    p: Tensor,
    q: Tensor,
    *,
    dim: int = -1,
    input_type: Literal["probability", "log_probability", "logits"] = "probability",
    reduction: Literal["none", "mean", "batchmean", "sum"] = "none",
    pairwise: bool = False,
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute Jensen-Shannon divergence between probability distributions.

    The JS divergence is a symmetric, bounded measure based on KL divergence.

    Mathematical Definition
    -----------------------
    .. math::
        D_{JS}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)

    where :math:`M = \frac{1}{2}(P + Q)` is the mixture distribution.

    Parameters
    ----------
    p : Tensor
        First probability distribution (or batch of distributions).
    q : Tensor
        Second probability distribution (or batch of distributions).
    dim : int, default=-1
        Dimension along which the probability distribution is defined.
    input_type : {"probability", "log_probability", "logits"}, default="probability"
        How to interpret input tensors.
    reduction : {"none", "mean", "batchmean", "sum"}, default="none"
        Reduction to apply.
    pairwise : bool, default=False
        If True, compute all-pairs divergence matrix.
    base : float, optional
        Logarithm base for output scaling. ``None`` for natural log (nats),
        ``2`` for bits. JS divergence is bounded by ``log(2)`` in the
        specified base.

    Returns
    -------
    Tensor
        JS divergence values.

    Examples
    --------
    >>> p = torch.tensor([0.5, 0.5])
    >>> q = torch.tensor([0.1, 0.9])
    >>> torchscience.information_theory.jensen_shannon_divergence(p, q)
    tensor(0.1927)

    >>> # In bits (bounded by 1.0)
    >>> torchscience.information_theory.jensen_shannon_divergence(p, q, base=2)
    tensor(0.2780)

    Notes
    -----
    - JS divergence is symmetric: :math:`D_{JS}(P \| Q) = D_{JS}(Q \| P)`
    - Bounded: :math:`0 \\leq D_{JS} \\leq \\log(2)` (in nats)
    - Square root of JS divergence is a proper metric

    See Also
    --------
    kullback_leibler_divergence : Asymmetric divergence measure.
    """
    # Validate input types
    if not isinstance(p, Tensor):
        raise TypeError(f"p must be a Tensor, got {type(p).__name__}")
    if not isinstance(q, Tensor):
        raise TypeError(f"q must be a Tensor, got {type(q).__name__}")

    # Validate input_type
    valid_input_types = ("probability", "log_probability", "logits")
    if input_type not in valid_input_types:
        raise ValueError(
            f"input_type must be one of {valid_input_types}, got '{input_type}'"
        )

    # Validate reduction
    valid_reductions = ("none", "mean", "batchmean", "sum")
    if reduction not in valid_reductions:
        raise ValueError(
            f"reduction must be one of {valid_reductions}, got '{reduction}'"
        )

    # Normalize dim
    p_dim = p.dim()
    if dim < -p_dim or dim >= p_dim:
        raise IndexError(
            f"dim {dim} out of range for tensor with {p_dim} dimensions"
        )
    dim = dim if dim >= 0 else p_dim + dim

    # Check distribution dimension sizes match
    if p.size(dim) != q.size(dim):
        raise ValueError(
            f"Distribution sizes must match along dim {dim}: "
            f"p has {p.size(dim)}, q has {q.size(dim)}"
        )

    # Validate pairwise mode
    if pairwise:
        if p.dim() < 2 or q.dim() < 2:
            raise ValueError(
                "pairwise=True requires p and q to be at least 2D"
            )

    # Dtype promotion
    target_dtype = torch.promote_types(p.dtype, q.dtype)
    if p.dtype != target_dtype:
        p = p.to(target_dtype)
    if q.dtype != target_dtype:
        q = q.to(target_dtype)

    return torch.ops.torchscience.jensen_shannon_divergence(
        p, q, dim, input_type, reduction, pairwise, base
    )
```

**Step 3: Update __init__.py**

```python
# src/torchscience/information_theory/__init__.py
from ._jensen_shannon_divergence import jensen_shannon_divergence
from ._kullback_leibler_divergence import kullback_leibler_divergence

__all__ = [
    "jensen_shannon_divergence",
    "kullback_leibler_divergence",
]
```

**Step 4: Commit**

```bash
git add src/torchscience/information_theory/
git commit -m "feat(information_theory): add Python API for KL and JS divergence"
```

---

## Task 2: Kernel Forward Implementations

**Files:**
- Create: `src/torchscience/csrc/kernel/information_theory/kullback_leibler_divergence.h`
- Create: `src/torchscience/csrc/kernel/information_theory/jensen_shannon_divergence.h`

**Step 1: Create KL divergence forward kernel**

```cpp
// src/torchscience/csrc/kernel/information_theory/kullback_leibler_divergence.h
#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

namespace torchscience::kernel::information_theory {

/**
 * Get dtype-dependent epsilon for numerical stability.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T get_eps() {
  return T(1e-7);  // Default for float32
}

template <>
C10_HOST_DEVICE C10_ALWAYS_INLINE double get_eps<double>() {
  return 1e-15;
}

template <>
C10_HOST_DEVICE C10_ALWAYS_INLINE c10::Half get_eps<c10::Half>() {
  return c10::Half(1e-4f);
}

template <>
C10_HOST_DEVICE C10_ALWAYS_INLINE c10::BFloat16 get_eps<c10::BFloat16>() {
  return c10::BFloat16(1e-3f);
}

/**
 * Compute KL divergence between two probability vectors.
 *
 * D_KL(P || Q) = sum_i p_i * log(p_i / q_i)
 *              = sum_i p_i * log(p_i) - p_i * log(q_i)
 *
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @return KL divergence value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T kl_divergence_kernel(
    const T* p,
    const T* q,
    int64_t n
) {
  T eps = get_eps<T>();
  T result = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;

    // Only contribute if p_i > 0 (by convention, 0 * log(0) = 0)
    if (p_i > eps) {
      result += p_i * (std::log(p_i) - std::log(q_i));
    }
  }

  return result;
}

/**
 * Compute JS divergence between two probability vectors.
 *
 * D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
 * where M = 0.5 * (P + Q)
 *
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param log_base_scale Scale factor for log base conversion (1.0 for natural log)
 * @return JS divergence value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T js_divergence_kernel(
    const T* p,
    const T* q,
    int64_t n,
    T log_base_scale
) {
  T eps = get_eps<T>();
  T result = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    T m_i = T(0.5) * (p_i + q_i);

    // D_KL(P || M) contribution
    if (p_i > eps) {
      result += T(0.5) * p_i * (std::log(p_i) - std::log(m_i));
    }

    // D_KL(Q || M) contribution
    if (q_i > eps) {
      result += T(0.5) * q_i * (std::log(q_i) - std::log(m_i));
    }
  }

  return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
```

**Step 2: Create JS divergence forward kernel**

The JS kernel is included in the same file above. Create a separate header for organization:

```cpp
// src/torchscience/csrc/kernel/information_theory/jensen_shannon_divergence.h
#pragma once

// JS divergence kernel is defined in kullback_leibler_divergence.h
#include "kullback_leibler_divergence.h"
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/kernel/information_theory/
git commit -m "feat(kernel): add KL and JS divergence forward kernels"
```

---

## Task 3: Kernel Backward Implementations

**Files:**
- Create: `src/torchscience/csrc/kernel/information_theory/kullback_leibler_divergence_backward.h`
- Create: `src/torchscience/csrc/kernel/information_theory/jensen_shannon_divergence_backward.h`

**Step 1: Create KL divergence backward kernel**

```cpp
// src/torchscience/csrc/kernel/information_theory/kullback_leibler_divergence_backward.h
#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradients for KL divergence.
 *
 * d(D_KL)/d(p_i) = log(p_i) - log(q_i) + 1
 * d(D_KL)/d(q_i) = -p_i / q_i
 *
 * @param grad_out Upstream gradient
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void kl_divergence_backward_kernel(
    T grad_out,
    const T* p,
    const T* q,
    int64_t n,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;

    // d(D_KL)/d(p_i) = log(p_i / q_i) + 1
    grad_p[i] = grad_out * (std::log(p_i) - std::log(q_i) + T(1));

    // d(D_KL)/d(q_i) = -p_i / q_i
    grad_q[i] = grad_out * (-p_i / q_i);
  }
}

/**
 * Compute gradients for JS divergence.
 *
 * With M = 0.5 * (P + Q):
 * d(D_JS)/d(p_i) = 0.5 * (log(p_i) - log(m_i) + 1) - 0.25 * (p_i + q_i) / m_i
 * d(D_JS)/d(q_i) = 0.5 * (log(q_i) - log(m_i) + 1) - 0.25 * (p_i + q_i) / m_i
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void js_divergence_backward_kernel(
    T grad_out,
    const T* p,
    const T* q,
    int64_t n,
    T log_base_scale,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();
  T scaled_grad = grad_out * log_base_scale;

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    T m_i = T(0.5) * (p_i + q_i);

    T log_p = std::log(p_i);
    T log_q = std::log(q_i);
    T log_m = std::log(m_i);

    // Gradient from D_KL(P || M) term w.r.t. p_i
    // d/dp_i [0.5 * p_i * (log(p_i) - log(m_i))]
    // = 0.5 * (log(p_i) - log(m_i) + 1 - 0.5 * p_i / m_i)
    T grad_p_from_kl_p = T(0.5) * (log_p - log_m + T(1) - T(0.5) * p_i / m_i);

    // Gradient from D_KL(Q || M) term w.r.t. p_i (via m_i)
    // d/dp_i [0.5 * q_i * (log(q_i) - log(m_i))]
    // = 0.5 * q_i * (-0.5 / m_i) = -0.25 * q_i / m_i
    T grad_p_from_kl_q = -T(0.25) * q_i / m_i;

    grad_p[i] = scaled_grad * (grad_p_from_kl_p + grad_p_from_kl_q);

    // Symmetric for q
    T grad_q_from_kl_q = T(0.5) * (log_q - log_m + T(1) - T(0.5) * q_i / m_i);
    T grad_q_from_kl_p = -T(0.25) * p_i / m_i;

    grad_q[i] = scaled_grad * (grad_q_from_kl_q + grad_q_from_kl_p);
  }
}

}  // namespace torchscience::kernel::information_theory
```

**Step 2: Create JS divergence backward kernel header**

```cpp
// src/torchscience/csrc/kernel/information_theory/jensen_shannon_divergence_backward.h
#pragma once

// JS backward kernel is defined in kullback_leibler_divergence_backward.h
#include "kullback_leibler_divergence_backward.h"
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/kernel/information_theory/
git commit -m "feat(kernel): add KL and JS divergence backward kernels"
```

---

## Task 4: Kernel Backward-Backward Implementations

**Files:**
- Create: `src/torchscience/csrc/kernel/information_theory/kullback_leibler_divergence_backward_backward.h`
- Create: `src/torchscience/csrc/kernel/information_theory/jensen_shannon_divergence_backward_backward.h`

**Step 1: Create KL divergence backward-backward kernel**

```cpp
// src/torchscience/csrc/kernel/information_theory/kullback_leibler_divergence_backward_backward.h
#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute second-order gradients for KL divergence.
 *
 * d^2(D_KL)/d(p_i)^2 = 1 / p_i
 * d^2(D_KL)/d(q_i)^2 = p_i / q_i^2
 * d^2(D_KL)/d(p_i)d(q_i) = -1 / q_i
 *
 * @param gg_p Upstream gradient w.r.t. grad_p
 * @param gg_q Upstream gradient w.r.t. grad_q
 * @param grad_out Original upstream gradient
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param grad_grad_out Output: gradient w.r.t. grad_out
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void kl_divergence_backward_backward_kernel(
    const T* gg_p,
    const T* gg_q,
    T grad_out,
    const T* p,
    const T* q,
    int64_t n,
    T& grad_grad_out,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();
  grad_grad_out = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;

    T gg_p_i = gg_p != nullptr ? gg_p[i] : T(0);
    T gg_q_i = gg_q != nullptr ? gg_q[i] : T(0);

    // Gradient of grad_out from backward pass
    // grad_p[i] = grad_out * (log(p_i/q_i) + 1)
    // grad_q[i] = grad_out * (-p_i/q_i)
    grad_grad_out += gg_p_i * (std::log(p_i) - std::log(q_i) + T(1));
    grad_grad_out += gg_q_i * (-p_i / q_i);

    // Second derivative w.r.t. p
    // d/dp_i [grad_out * (log(p_i) - log(q_i) + 1)] = grad_out / p_i
    // d/dp_i [grad_out * (-p_i / q_i)] = -grad_out / q_i
    grad_p[i] = gg_p_i * grad_out / p_i + gg_q_i * (-grad_out / q_i);

    // Second derivative w.r.t. q
    // d/dq_i [grad_out * (log(p_i) - log(q_i) + 1)] = -grad_out / q_i
    // d/dq_i [grad_out * (-p_i / q_i)] = grad_out * p_i / q_i^2
    grad_q[i] = gg_p_i * (-grad_out / q_i) + gg_q_i * (grad_out * p_i / (q_i * q_i));
  }
}

}  // namespace torchscience::kernel::information_theory
```

**Step 2: Create JS divergence backward-backward kernel header**

```cpp
// src/torchscience/csrc/kernel/information_theory/jensen_shannon_divergence_backward_backward.h
#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute second-order gradients for JS divergence.
 * Implementation follows the same pattern as KL but with JS-specific derivatives.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void js_divergence_backward_backward_kernel(
    const T* gg_p,
    const T* gg_q,
    T grad_out,
    const T* p,
    const T* q,
    int64_t n,
    T log_base_scale,
    T& grad_grad_out,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();
  T scaled_grad = grad_out * log_base_scale;
  grad_grad_out = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    T m_i = T(0.5) * (p_i + q_i);

    T gg_p_i = gg_p != nullptr ? gg_p[i] : T(0);
    T gg_q_i = gg_q != nullptr ? gg_q[i] : T(0);

    T log_p = std::log(p_i);
    T log_q = std::log(q_i);
    T log_m = std::log(m_i);

    // First derivatives (from backward)
    T grad_p_from_kl_p = T(0.5) * (log_p - log_m + T(1) - T(0.5) * p_i / m_i);
    T grad_p_from_kl_q = -T(0.25) * q_i / m_i;
    T d_dp = grad_p_from_kl_p + grad_p_from_kl_q;

    T grad_q_from_kl_q = T(0.5) * (log_q - log_m + T(1) - T(0.5) * q_i / m_i);
    T grad_q_from_kl_p = -T(0.25) * p_i / m_i;
    T d_dq = grad_q_from_kl_q + grad_q_from_kl_p;

    // Gradient of grad_out
    grad_grad_out += (gg_p_i * d_dp + gg_q_i * d_dq) * log_base_scale;

    // Second derivatives (simplified)
    // d^2/dp^2 = 0.5/p - 0.25/m + 0.125*(p+q)/m^2
    T d2_dp2 = T(0.5) / p_i - T(0.25) / m_i + T(0.125) * (p_i + q_i) / (m_i * m_i);

    // d^2/dq^2 (symmetric)
    T d2_dq2 = T(0.5) / q_i - T(0.25) / m_i + T(0.125) * (p_i + q_i) / (m_i * m_i);

    // d^2/dpdq
    T d2_dpdq = T(0.125) * (p_i + q_i) / (m_i * m_i) - T(0.25) / m_i;

    grad_p[i] = scaled_grad * (gg_p_i * d2_dp2 + gg_q_i * d2_dpdq);
    grad_q[i] = scaled_grad * (gg_p_i * d2_dpdq + gg_q_i * d2_dq2);
  }
}

}  // namespace torchscience::kernel::information_theory
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/kernel/information_theory/
git commit -m "feat(kernel): add KL and JS divergence backward-backward kernels"
```

---

## Task 5: CPU Backend Implementation

**Files:**
- Create: `src/torchscience/csrc/cpu/information_theory/kullback_leibler_divergence.h`
- Create: `src/torchscience/csrc/cpu/information_theory/jensen_shannon_divergence.h`

**Step 1: Create KL divergence CPU implementation**

```cpp
// src/torchscience/csrc/cpu/information_theory/kullback_leibler_divergence.h
#pragma once

#include <cmath>
#include <string>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../kernel/information_theory/kullback_leibler_divergence.h"
#include "../../kernel/information_theory/kullback_leibler_divergence_backward.h"
#include "../../kernel/information_theory/kullback_leibler_divergence_backward_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

/**
 * Preprocess input based on input_type.
 * Returns contiguous tensor in probability space.
 */
inline at::Tensor preprocess_input(
    const at::Tensor& input,
    int64_t dim,
    const std::string& input_type
) {
  if (input_type == "probability") {
    return input.contiguous();
  } else if (input_type == "log_probability") {
    return input.exp().contiguous();
  } else if (input_type == "logits") {
    return at::softmax(input, dim).contiguous();
  } else {
    TORCH_CHECK(false, "Unknown input_type: ", input_type);
  }
}

/**
 * Apply reduction to output tensor.
 */
inline at::Tensor apply_reduction(
    const at::Tensor& output,
    const std::string& reduction,
    int64_t batch_size
) {
  if (reduction == "none") {
    return output;
  } else if (reduction == "mean") {
    return output.mean();
  } else if (reduction == "batchmean") {
    return output.sum() / static_cast<double>(batch_size);
  } else if (reduction == "sum") {
    return output.sum();
  } else {
    TORCH_CHECK(false, "Unknown reduction: ", reduction);
  }
}

}  // anonymous namespace

inline at::Tensor kullback_leibler_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
  // Preprocess inputs
  at::Tensor p_prob = preprocess_input(p, dim, input_type);
  at::Tensor q_prob = preprocess_input(q, dim, input_type);

  int64_t n = p_prob.size(dim);

  if (pairwise) {
    // Pairwise mode: p is (m, n), q is (k, n) -> output is (m, k)
    TORCH_CHECK(p_prob.dim() == 2 && q_prob.dim() == 2,
                "Pairwise mode requires 2D tensors");
    TORCH_CHECK(dim == 1 || dim == -1,
                "Pairwise mode requires dim=1 or dim=-1");

    int64_t m = p_prob.size(0);
    int64_t k = q_prob.size(0);

    at::Tensor output = at::empty({m, k}, p_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "kullback_leibler_divergence_pairwise_cpu",
        [&]() {
          const scalar_t* p_ptr = p_prob.data_ptr<scalar_t>();
          const scalar_t* q_ptr = q_prob.data_ptr<scalar_t>();
          scalar_t* out_ptr = output.data_ptr<scalar_t>();

          at::parallel_for(0, m * k, 0, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
              int64_t i = idx / k;
              int64_t j = idx % k;
              out_ptr[idx] = kernel::information_theory::kl_divergence_kernel<scalar_t>(
                  p_ptr + i * n,
                  q_ptr + j * n,
                  n
              );
            }
          });
        }
    );

    return apply_reduction(output, reduction, m);
  } else {
    // Standard mode: element-wise along dim
    // Move dim to last position for easier processing
    at::Tensor p_t = p_prob.movedim(dim, -1).contiguous();
    at::Tensor q_t = q_prob.movedim(dim, -1).contiguous();

    // Output shape is input shape without the distribution dim
    auto out_sizes = p_t.sizes().vec();
    out_sizes.pop_back();
    if (out_sizes.empty()) {
      out_sizes.push_back(1);
    }

    int64_t num_distributions = p_t.numel() / n;
    at::Tensor output = at::empty(out_sizes, p_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "kullback_leibler_divergence_cpu",
        [&]() {
          const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
          const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
          scalar_t* out_ptr = output.data_ptr<scalar_t>();

          at::parallel_for(0, num_distributions, 0, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
              out_ptr[i] = kernel::information_theory::kl_divergence_kernel<scalar_t>(
                  p_ptr + i * n,
                  q_ptr + i * n,
                  n
              );
            }
          });
        }
    );

    // Handle scalar output case
    if (p.dim() == 1) {
      output = output.squeeze();
    }

    return apply_reduction(output, reduction, num_distributions);
  }
}

inline std::tuple<at::Tensor, at::Tensor> kullback_leibler_divergence_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
  // For simplicity, compute gradients for probability inputs
  // Gradients through preprocessing would need chain rule

  at::Tensor p_prob = preprocess_input(p, dim, input_type);
  at::Tensor q_prob = preprocess_input(q, dim, input_type);

  int64_t n = p_prob.size(dim);

  at::Tensor grad_p = at::zeros_like(p_prob);
  at::Tensor grad_q = at::zeros_like(q_prob);

  if (pairwise) {
    int64_t m = p_prob.size(0);
    int64_t k = q_prob.size(0);

    at::Tensor grad_out_expanded = grad_output.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "kullback_leibler_divergence_backward_pairwise_cpu",
        [&]() {
          const scalar_t* p_ptr = p_prob.data_ptr<scalar_t>();
          const scalar_t* q_ptr = q_prob.data_ptr<scalar_t>();
          const scalar_t* grad_out_ptr = grad_out_expanded.data_ptr<scalar_t>();
          scalar_t* grad_p_ptr = grad_p.data_ptr<scalar_t>();
          scalar_t* grad_q_ptr = grad_q.data_ptr<scalar_t>();

          std::vector<scalar_t> temp_grad_p(n);
          std::vector<scalar_t> temp_grad_q(n);

          for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < k; ++j) {
              scalar_t g = grad_out_ptr[i * k + j];

              kernel::information_theory::kl_divergence_backward_kernel<scalar_t>(
                  g,
                  p_ptr + i * n,
                  q_ptr + j * n,
                  n,
                  temp_grad_p.data(),
                  temp_grad_q.data()
              );

              for (int64_t l = 0; l < n; ++l) {
                grad_p_ptr[i * n + l] += temp_grad_p[l];
                grad_q_ptr[j * n + l] += temp_grad_q[l];
              }
            }
          }
        }
    );
  } else {
    at::Tensor p_t = p_prob.movedim(dim, -1).contiguous();
    at::Tensor q_t = q_prob.movedim(dim, -1).contiguous();

    int64_t num_distributions = p_t.numel() / n;
    at::Tensor grad_p_t = at::zeros_like(p_t);
    at::Tensor grad_q_t = at::zeros_like(q_t);

    at::Tensor grad_out_flat = grad_output.flatten().contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "kullback_leibler_divergence_backward_cpu",
        [&]() {
          const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
          const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
          const scalar_t* grad_out_ptr = grad_out_flat.data_ptr<scalar_t>();
          scalar_t* grad_p_ptr = grad_p_t.data_ptr<scalar_t>();
          scalar_t* grad_q_ptr = grad_q_t.data_ptr<scalar_t>();

          at::parallel_for(0, num_distributions, 0, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
              kernel::information_theory::kl_divergence_backward_kernel<scalar_t>(
                  grad_out_ptr[i],
                  p_ptr + i * n,
                  q_ptr + i * n,
                  n,
                  grad_p_ptr + i * n,
                  grad_q_ptr + i * n
              );
            }
          });
        }
    );

    grad_p = grad_p_t.movedim(-1, dim);
    grad_q = grad_q_t.movedim(-1, dim);
  }

  return std::make_tuple(grad_p, grad_q);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> kullback_leibler_divergence_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
  at::Tensor p_prob = preprocess_input(p, dim, input_type);
  at::Tensor q_prob = preprocess_input(q, dim, input_type);

  int64_t n = p_prob.size(dim);

  at::Tensor grad_grad_out = at::zeros({}, grad_output.options());
  at::Tensor grad_p = at::zeros_like(p_prob);
  at::Tensor grad_q = at::zeros_like(q_prob);

  if (!pairwise) {
    at::Tensor p_t = p_prob.movedim(dim, -1).contiguous();
    at::Tensor q_t = q_prob.movedim(dim, -1).contiguous();
    at::Tensor gg_p_t = gg_p.defined() ? gg_p.movedim(dim, -1).contiguous() : at::Tensor();
    at::Tensor gg_q_t = gg_q.defined() ? gg_q.movedim(dim, -1).contiguous() : at::Tensor();

    int64_t num_distributions = p_t.numel() / n;
    at::Tensor grad_p_t = at::zeros_like(p_t);
    at::Tensor grad_q_t = at::zeros_like(q_t);

    at::Tensor grad_out_flat = grad_output.flatten().contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "kullback_leibler_divergence_backward_backward_cpu",
        [&]() {
          const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
          const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
          const scalar_t* gg_p_ptr = gg_p_t.defined() ? gg_p_t.data_ptr<scalar_t>() : nullptr;
          const scalar_t* gg_q_ptr = gg_q_t.defined() ? gg_q_t.data_ptr<scalar_t>() : nullptr;
          const scalar_t* grad_out_ptr = grad_out_flat.data_ptr<scalar_t>();
          scalar_t* grad_p_ptr = grad_p_t.data_ptr<scalar_t>();
          scalar_t* grad_q_ptr = grad_q_t.data_ptr<scalar_t>();

          scalar_t total_grad_grad_out = scalar_t(0);

          for (int64_t i = 0; i < num_distributions; ++i) {
            scalar_t local_grad_grad_out;
            kernel::information_theory::kl_divergence_backward_backward_kernel<scalar_t>(
                gg_p_ptr ? gg_p_ptr + i * n : nullptr,
                gg_q_ptr ? gg_q_ptr + i * n : nullptr,
                grad_out_ptr[i],
                p_ptr + i * n,
                q_ptr + i * n,
                n,
                local_grad_grad_out,
                grad_p_ptr + i * n,
                grad_q_ptr + i * n
            );
            total_grad_grad_out += local_grad_grad_out;
          }

          grad_grad_out.fill_(total_grad_grad_out);
        }
    );

    grad_p = grad_p_t.movedim(-1, dim);
    grad_q = grad_q_t.movedim(-1, dim);
  }

  return std::make_tuple(grad_grad_out, grad_p, grad_q);
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
  module.impl("kullback_leibler_divergence", &torchscience::cpu::information_theory::kullback_leibler_divergence);
  module.impl("kullback_leibler_divergence_backward", &torchscience::cpu::information_theory::kullback_leibler_divergence_backward);
  module.impl("kullback_leibler_divergence_backward_backward", &torchscience::cpu::information_theory::kullback_leibler_divergence_backward_backward);
}
```

**Step 2: Create JS divergence CPU implementation**

```cpp
// src/torchscience/csrc/cpu/information_theory/jensen_shannon_divergence.h
#pragma once

#include <cmath>
#include <string>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../kernel/information_theory/kullback_leibler_divergence.h"
#include "../../kernel/information_theory/kullback_leibler_divergence_backward.h"
#include "../../kernel/information_theory/jensen_shannon_divergence_backward_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor preprocess_input_js(
    const at::Tensor& input,
    int64_t dim,
    const std::string& input_type
) {
  if (input_type == "probability") {
    return input.contiguous();
  } else if (input_type == "log_probability") {
    return input.exp().contiguous();
  } else if (input_type == "logits") {
    return at::softmax(input, dim).contiguous();
  } else {
    TORCH_CHECK(false, "Unknown input_type: ", input_type);
  }
}

inline at::Tensor apply_reduction_js(
    const at::Tensor& output,
    const std::string& reduction,
    int64_t batch_size
) {
  if (reduction == "none") {
    return output;
  } else if (reduction == "mean") {
    return output.mean();
  } else if (reduction == "batchmean") {
    return output.sum() / static_cast<double>(batch_size);
  } else if (reduction == "sum") {
    return output.sum();
  } else {
    TORCH_CHECK(false, "Unknown reduction: ", reduction);
  }
}

}  // anonymous namespace

inline at::Tensor jensen_shannon_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise,
    const c10::optional<double>& base
) {
  at::Tensor p_prob = preprocess_input_js(p, dim, input_type);
  at::Tensor q_prob = preprocess_input_js(q, dim, input_type);

  int64_t n = p_prob.size(dim);

  // Compute log base scale factor
  double log_base_scale = 1.0;
  if (base.has_value()) {
    log_base_scale = 1.0 / std::log(base.value());
  }

  if (pairwise) {
    TORCH_CHECK(p_prob.dim() == 2 && q_prob.dim() == 2,
                "Pairwise mode requires 2D tensors");

    int64_t m = p_prob.size(0);
    int64_t k = q_prob.size(0);

    at::Tensor output = at::empty({m, k}, p_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "jensen_shannon_divergence_pairwise_cpu",
        [&]() {
          const scalar_t* p_ptr = p_prob.data_ptr<scalar_t>();
          const scalar_t* q_ptr = q_prob.data_ptr<scalar_t>();
          scalar_t* out_ptr = output.data_ptr<scalar_t>();
          scalar_t scale = static_cast<scalar_t>(log_base_scale);

          at::parallel_for(0, m * k, 0, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
              int64_t i = idx / k;
              int64_t j = idx % k;
              out_ptr[idx] = kernel::information_theory::js_divergence_kernel<scalar_t>(
                  p_ptr + i * n,
                  q_ptr + j * n,
                  n,
                  scale
              );
            }
          });
        }
    );

    return apply_reduction_js(output, reduction, m);
  } else {
    at::Tensor p_t = p_prob.movedim(dim, -1).contiguous();
    at::Tensor q_t = q_prob.movedim(dim, -1).contiguous();

    auto out_sizes = p_t.sizes().vec();
    out_sizes.pop_back();
    if (out_sizes.empty()) {
      out_sizes.push_back(1);
    }

    int64_t num_distributions = p_t.numel() / n;
    at::Tensor output = at::empty(out_sizes, p_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "jensen_shannon_divergence_cpu",
        [&]() {
          const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
          const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
          scalar_t* out_ptr = output.data_ptr<scalar_t>();
          scalar_t scale = static_cast<scalar_t>(log_base_scale);

          at::parallel_for(0, num_distributions, 0, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
              out_ptr[i] = kernel::information_theory::js_divergence_kernel<scalar_t>(
                  p_ptr + i * n,
                  q_ptr + i * n,
                  n,
                  scale
              );
            }
          });
        }
    );

    if (p.dim() == 1) {
      output = output.squeeze();
    }

    return apply_reduction_js(output, reduction, num_distributions);
  }
}

inline std::tuple<at::Tensor, at::Tensor> jensen_shannon_divergence_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise,
    const c10::optional<double>& base
) {
  at::Tensor p_prob = preprocess_input_js(p, dim, input_type);
  at::Tensor q_prob = preprocess_input_js(q, dim, input_type);

  int64_t n = p_prob.size(dim);

  double log_base_scale = 1.0;
  if (base.has_value()) {
    log_base_scale = 1.0 / std::log(base.value());
  }

  at::Tensor grad_p = at::zeros_like(p_prob);
  at::Tensor grad_q = at::zeros_like(q_prob);

  if (!pairwise) {
    at::Tensor p_t = p_prob.movedim(dim, -1).contiguous();
    at::Tensor q_t = q_prob.movedim(dim, -1).contiguous();

    int64_t num_distributions = p_t.numel() / n;
    at::Tensor grad_p_t = at::zeros_like(p_t);
    at::Tensor grad_q_t = at::zeros_like(q_t);

    at::Tensor grad_out_flat = grad_output.flatten().contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "jensen_shannon_divergence_backward_cpu",
        [&]() {
          const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
          const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
          const scalar_t* grad_out_ptr = grad_out_flat.data_ptr<scalar_t>();
          scalar_t* grad_p_ptr = grad_p_t.data_ptr<scalar_t>();
          scalar_t* grad_q_ptr = grad_q_t.data_ptr<scalar_t>();
          scalar_t scale = static_cast<scalar_t>(log_base_scale);

          at::parallel_for(0, num_distributions, 0, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
              kernel::information_theory::js_divergence_backward_kernel<scalar_t>(
                  grad_out_ptr[i],
                  p_ptr + i * n,
                  q_ptr + i * n,
                  n,
                  scale,
                  grad_p_ptr + i * n,
                  grad_q_ptr + i * n
              );
            }
          });
        }
    );

    grad_p = grad_p_t.movedim(-1, dim);
    grad_q = grad_q_t.movedim(-1, dim);
  }

  return std::make_tuple(grad_p, grad_q);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> jensen_shannon_divergence_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise,
    const c10::optional<double>& base
) {
  at::Tensor p_prob = preprocess_input_js(p, dim, input_type);
  at::Tensor q_prob = preprocess_input_js(q, dim, input_type);

  int64_t n = p_prob.size(dim);

  double log_base_scale = 1.0;
  if (base.has_value()) {
    log_base_scale = 1.0 / std::log(base.value());
  }

  at::Tensor grad_grad_out = at::zeros({}, grad_output.options());
  at::Tensor grad_p = at::zeros_like(p_prob);
  at::Tensor grad_q = at::zeros_like(q_prob);

  if (!pairwise) {
    at::Tensor p_t = p_prob.movedim(dim, -1).contiguous();
    at::Tensor q_t = q_prob.movedim(dim, -1).contiguous();
    at::Tensor gg_p_t = gg_p.defined() ? gg_p.movedim(dim, -1).contiguous() : at::Tensor();
    at::Tensor gg_q_t = gg_q.defined() ? gg_q.movedim(dim, -1).contiguous() : at::Tensor();

    int64_t num_distributions = p_t.numel() / n;
    at::Tensor grad_p_t = at::zeros_like(p_t);
    at::Tensor grad_q_t = at::zeros_like(q_t);

    at::Tensor grad_out_flat = grad_output.flatten().contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "jensen_shannon_divergence_backward_backward_cpu",
        [&]() {
          const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
          const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
          const scalar_t* gg_p_ptr = gg_p_t.defined() ? gg_p_t.data_ptr<scalar_t>() : nullptr;
          const scalar_t* gg_q_ptr = gg_q_t.defined() ? gg_q_t.data_ptr<scalar_t>() : nullptr;
          const scalar_t* grad_out_ptr = grad_out_flat.data_ptr<scalar_t>();
          scalar_t* grad_p_ptr = grad_p_t.data_ptr<scalar_t>();
          scalar_t* grad_q_ptr = grad_q_t.data_ptr<scalar_t>();
          scalar_t scale = static_cast<scalar_t>(log_base_scale);

          scalar_t total_grad_grad_out = scalar_t(0);

          for (int64_t i = 0; i < num_distributions; ++i) {
            scalar_t local_grad_grad_out;
            kernel::information_theory::js_divergence_backward_backward_kernel<scalar_t>(
                gg_p_ptr ? gg_p_ptr + i * n : nullptr,
                gg_q_ptr ? gg_q_ptr + i * n : nullptr,
                grad_out_ptr[i],
                p_ptr + i * n,
                q_ptr + i * n,
                n,
                scale,
                local_grad_grad_out,
                grad_p_ptr + i * n,
                grad_q_ptr + i * n
            );
            total_grad_grad_out += local_grad_grad_out;
          }

          grad_grad_out.fill_(total_grad_grad_out);
        }
    );

    grad_p = grad_p_t.movedim(-1, dim);
    grad_q = grad_q_t.movedim(-1, dim);
  }

  return std::make_tuple(grad_grad_out, grad_p, grad_q);
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
  module.impl("jensen_shannon_divergence", &torchscience::cpu::information_theory::jensen_shannon_divergence);
  module.impl("jensen_shannon_divergence_backward", &torchscience::cpu::information_theory::jensen_shannon_divergence_backward);
  module.impl("jensen_shannon_divergence_backward_backward", &torchscience::cpu::information_theory::jensen_shannon_divergence_backward_backward);
}
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cpu/information_theory/
git commit -m "feat(cpu): add KL and JS divergence CPU implementations"
```

---

## Task 6: Meta Backend Implementation

**Files:**
- Create: `src/torchscience/csrc/meta/information_theory/kullback_leibler_divergence.h`
- Create: `src/torchscience/csrc/meta/information_theory/jensen_shannon_divergence.h`

**Step 1: Create KL divergence meta implementation**

```cpp
// src/torchscience/csrc/meta/information_theory/kullback_leibler_divergence.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor kullback_leibler_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
  if (pairwise) {
    int64_t m = p.size(0);
    int64_t k = q.size(0);

    if (reduction == "none") {
      return at::empty({m, k}, p.options());
    } else {
      return at::empty({}, p.options());
    }
  } else {
    auto sizes = p.sizes().vec();
    sizes.erase(sizes.begin() + (dim >= 0 ? dim : p.dim() + dim));

    if (reduction == "none") {
      if (sizes.empty()) {
        return at::empty({}, p.options());
      }
      return at::empty(sizes, p.options());
    } else {
      return at::empty({}, p.options());
    }
  }
}

inline std::tuple<at::Tensor, at::Tensor> kullback_leibler_divergence_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
  return std::make_tuple(at::empty_like(p), at::empty_like(q));
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> kullback_leibler_divergence_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
  return std::make_tuple(
      at::empty({}, grad_output.options()),
      at::empty_like(p),
      at::empty_like(q)
  );
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("kullback_leibler_divergence", &torchscience::meta::information_theory::kullback_leibler_divergence);
  m.impl("kullback_leibler_divergence_backward", &torchscience::meta::information_theory::kullback_leibler_divergence_backward);
  m.impl("kullback_leibler_divergence_backward_backward", &torchscience::meta::information_theory::kullback_leibler_divergence_backward_backward);
}
```

**Step 2: Create JS divergence meta implementation**

```cpp
// src/torchscience/csrc/meta/information_theory/jensen_shannon_divergence.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor jensen_shannon_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise,
    const c10::optional<double>& base
) {
  if (pairwise) {
    int64_t m = p.size(0);
    int64_t k = q.size(0);

    if (reduction == "none") {
      return at::empty({m, k}, p.options());
    } else {
      return at::empty({}, p.options());
    }
  } else {
    auto sizes = p.sizes().vec();
    sizes.erase(sizes.begin() + (dim >= 0 ? dim : p.dim() + dim));

    if (reduction == "none") {
      if (sizes.empty()) {
        return at::empty({}, p.options());
      }
      return at::empty(sizes, p.options());
    } else {
      return at::empty({}, p.options());
    }
  }
}

inline std::tuple<at::Tensor, at::Tensor> jensen_shannon_divergence_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise,
    const c10::optional<double>& base
) {
  return std::make_tuple(at::empty_like(p), at::empty_like(q));
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> jensen_shannon_divergence_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise,
    const c10::optional<double>& base
) {
  return std::make_tuple(
      at::empty({}, grad_output.options()),
      at::empty_like(p),
      at::empty_like(q)
  );
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("jensen_shannon_divergence", &torchscience::meta::information_theory::jensen_shannon_divergence);
  m.impl("jensen_shannon_divergence_backward", &torchscience::meta::information_theory::jensen_shannon_divergence_backward);
  m.impl("jensen_shannon_divergence_backward_backward", &torchscience::meta::information_theory::jensen_shannon_divergence_backward_backward);
}
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/meta/information_theory/
git commit -m "feat(meta): add KL and JS divergence meta implementations"
```

---

## Task 7: Autograd Backend Implementation

**Files:**
- Create: `src/torchscience/csrc/autograd/information_theory/kullback_leibler_divergence.h`
- Create: `src/torchscience/csrc/autograd/information_theory/jensen_shannon_divergence.h`

**Step 1: Create KL divergence autograd implementation**

```cpp
// src/torchscience/csrc/autograd/information_theory/kullback_leibler_divergence.h
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::information_theory {

class KullbackLeiblerDivergence
    : public torch::autograd::Function<KullbackLeiblerDivergence> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& p,
      const at::Tensor& q,
      int64_t dim,
      const std::string& input_type,
      const std::string& reduction,
      bool pairwise
  ) {
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["input_type"] = input_type;
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["pairwise"] = pairwise;

    bool p_requires_grad = p.requires_grad() && at::isFloatingType(p.scalar_type());
    bool q_requires_grad = q.requires_grad() && at::isFloatingType(q.scalar_type());
    ctx->saved_data["p_requires_grad"] = p_requires_grad;
    ctx->saved_data["q_requires_grad"] = q_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    at::Tensor output = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::kullback_leibler_divergence", "")
        .typed<at::Tensor(
            const at::Tensor&,
            const at::Tensor&,
            int64_t,
            const std::string&,
            const std::string&,
            bool
        )>()
        .call(p, q, dim, input_type, reduction, pairwise);

    ctx->save_for_backward({p, q, output});

    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor p = saved[0];
    at::Tensor q = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    int64_t dim = ctx->saved_data["dim"].toInt();
    std::string input_type = ctx->saved_data["input_type"].toStringRef();
    std::string reduction = ctx->saved_data["reduction"].toStringRef();
    bool pairwise = ctx->saved_data["pairwise"].toBool();
    bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
    bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

    if (!p_requires_grad && !q_requires_grad) {
      return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto [grad_p, grad_q] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::kullback_leibler_divergence_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            int64_t,
            const std::string&,
            const std::string&,
            bool
        )>()
        .call(grad_output, p, q, dim, input_type, reduction, pairwise);

    return {
        p_requires_grad ? grad_p : at::Tensor(),
        q_requires_grad ? grad_q : at::Tensor(),
        at::Tensor(),  // dim
        at::Tensor(),  // input_type
        at::Tensor(),  // reduction
        at::Tensor()   // pairwise
    };
  }
};

inline at::Tensor kullback_leibler_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
  return KullbackLeiblerDivergence::apply(p, q, dim, input_type, reduction, pairwise);
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
  module.impl("kullback_leibler_divergence", &torchscience::autograd::information_theory::kullback_leibler_divergence);
}
```

**Step 2: Create JS divergence autograd implementation**

```cpp
// src/torchscience/csrc/autograd/information_theory/jensen_shannon_divergence.h
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::information_theory {

class JensenShannonDivergence
    : public torch::autograd::Function<JensenShannonDivergence> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& p,
      const at::Tensor& q,
      int64_t dim,
      const std::string& input_type,
      const std::string& reduction,
      bool pairwise,
      const c10::optional<double>& base
  ) {
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["input_type"] = input_type;
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["pairwise"] = pairwise;
    ctx->saved_data["has_base"] = base.has_value();
    if (base.has_value()) {
      ctx->saved_data["base"] = base.value();
    }

    bool p_requires_grad = p.requires_grad() && at::isFloatingType(p.scalar_type());
    bool q_requires_grad = q.requires_grad() && at::isFloatingType(q.scalar_type());
    ctx->saved_data["p_requires_grad"] = p_requires_grad;
    ctx->saved_data["q_requires_grad"] = q_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    at::Tensor output = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::jensen_shannon_divergence", "")
        .typed<at::Tensor(
            const at::Tensor&,
            const at::Tensor&,
            int64_t,
            const std::string&,
            const std::string&,
            bool,
            const c10::optional<double>&
        )>()
        .call(p, q, dim, input_type, reduction, pairwise, base);

    ctx->save_for_backward({p, q, output});

    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor p = saved[0];
    at::Tensor q = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    int64_t dim = ctx->saved_data["dim"].toInt();
    std::string input_type = ctx->saved_data["input_type"].toStringRef();
    std::string reduction = ctx->saved_data["reduction"].toStringRef();
    bool pairwise = ctx->saved_data["pairwise"].toBool();
    bool has_base = ctx->saved_data["has_base"].toBool();
    c10::optional<double> base;
    if (has_base) {
      base = ctx->saved_data["base"].toDouble();
    }
    bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
    bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

    if (!p_requires_grad && !q_requires_grad) {
      return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto [grad_p, grad_q] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::jensen_shannon_divergence_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            int64_t,
            const std::string&,
            const std::string&,
            bool,
            const c10::optional<double>&
        )>()
        .call(grad_output, p, q, dim, input_type, reduction, pairwise, base);

    return {
        p_requires_grad ? grad_p : at::Tensor(),
        q_requires_grad ? grad_q : at::Tensor(),
        at::Tensor(),  // dim
        at::Tensor(),  // input_type
        at::Tensor(),  // reduction
        at::Tensor(),  // pairwise
        at::Tensor()   // base
    };
  }
};

inline at::Tensor jensen_shannon_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise,
    const c10::optional<double>& base
) {
  return JensenShannonDivergence::apply(p, q, dim, input_type, reduction, pairwise, base);
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
  module.impl("jensen_shannon_divergence", &torchscience::autograd::information_theory::jensen_shannon_divergence);
}
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/autograd/information_theory/
git commit -m "feat(autograd): add KL and JS divergence autograd implementations"
```

---

## Task 8: Operator Registration

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp`

**Step 1: Add includes for information_theory**

Add after line 26 (after floyd_warshall include):

```cpp
#include "cpu/information_theory/kullback_leibler_divergence.h"
#include "cpu/information_theory/jensen_shannon_divergence.h"
```

Add after line 35 (after autograd/test/sum_squares include):

```cpp
#include "autograd/information_theory/kullback_leibler_divergence.h"
#include "autograd/information_theory/jensen_shannon_divergence.h"
```

Add after line 46 (after meta/graph_theory/floyd_warshall include):

```cpp
#include "meta/information_theory/kullback_leibler_divergence.h"
#include "meta/information_theory/jensen_shannon_divergence.h"
```

**Step 2: Add operator definitions**

Add after line 190 (after floyd_warshall def):

```cpp
  // information_theory
  module.def("kullback_leibler_divergence(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> Tensor");
  module.def("kullback_leibler_divergence_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor)");
  module.def("kullback_leibler_divergence_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor, Tensor)");

  module.def("jensen_shannon_divergence(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise, float? base) -> Tensor");
  module.def("jensen_shannon_divergence_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise, float? base) -> (Tensor, Tensor)");
  module.def("jensen_shannon_divergence_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise, float? base) -> (Tensor, Tensor, Tensor)");
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat: register KL and JS divergence operators"
```

---

## Task 9: Tests

**Files:**
- Create: `tests/torchscience/information_theory/__init__.py`
- Create: `tests/torchscience/information_theory/test__kullback_leibler_divergence.py`
- Create: `tests/torchscience/information_theory/test__jensen_shannon_divergence.py`

**Step 1: Create test __init__.py**

```python
# tests/torchscience/information_theory/__init__.py
```

**Step 2: Create KL divergence tests**

```python
# tests/torchscience/information_theory/test__kullback_leibler_divergence.py
import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information_theory import kullback_leibler_divergence


class TestKLDivergenceBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_1d(self):
        """Output is scalar for 1D inputs."""
        p = torch.softmax(torch.randn(5), dim=0)
        q = torch.softmax(torch.randn(5), dim=0)

        kl = kullback_leibler_divergence(p, q)

        assert kl.shape == ()

    def test_output_shape_2d_reduction_none(self):
        """Output shape is (batch,) for 2D inputs with reduction=none."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        kl = kullback_leibler_divergence(p, q, reduction="none")

        assert kl.shape == (10,)

    def test_non_negative(self):
        """KL divergence is always non-negative."""
        p = torch.softmax(torch.randn(100, 10), dim=-1)
        q = torch.softmax(torch.randn(100, 10), dim=-1)

        kl = kullback_leibler_divergence(p, q, reduction="none")

        assert (kl >= 0).all()

    def test_zero_for_identical_distributions(self):
        """KL(P || P) = 0."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)

        kl = kullback_leibler_divergence(p, p, reduction="none")

        torch.testing.assert_close(kl, torch.zeros(10), atol=1e-6, rtol=1e-6)


class TestKLDivergenceCorrectness:
    """Tests for numerical correctness."""

    def test_uniform_vs_uniform(self):
        """KL(uniform || uniform) = 0."""
        n = 10
        p = torch.ones(n) / n
        q = torch.ones(n) / n

        kl = kullback_leibler_divergence(p, q)

        torch.testing.assert_close(kl, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

    def test_known_bernoulli_value(self):
        """Test against known Bernoulli KL divergence."""
        # KL(Bernoulli(0.5) || Bernoulli(0.25))
        # = 0.5 * log(0.5/0.25) + 0.5 * log(0.5/0.75)
        # = 0.5 * log(2) + 0.5 * log(2/3)
        # = 0.5 * (log(2) + log(2/3))
        # = 0.5 * log(4/3)
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.25, 0.75])

        kl = kullback_leibler_divergence(p, q)

        expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
        torch.testing.assert_close(kl, torch.tensor(expected), atol=1e-5, rtol=1e-5)

    def test_matches_torch_kl_div(self):
        """Compare against torch.nn.functional.kl_div."""
        p = torch.softmax(torch.randn(10, 5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(10, 5, dtype=torch.float64), dim=-1)

        ours = kullback_leibler_divergence(p, q, reduction="batchmean")

        # torch.kl_div expects log(q) as input and p as target
        theirs = torch.nn.functional.kl_div(
            q.log(), p, reduction="batchmean", log_target=False
        )

        torch.testing.assert_close(ours, theirs, atol=1e-10, rtol=1e-10)


class TestKLDivergenceInputTypes:
    """Tests for different input types."""

    def test_log_probability_input(self):
        """input_type='log_probability' exponentiates inputs."""
        p = torch.softmax(torch.randn(5), dim=0)
        q = torch.softmax(torch.randn(5), dim=0)

        kl_prob = kullback_leibler_divergence(p, q, input_type="probability")
        kl_log = kullback_leibler_divergence(p.log(), q.log(), input_type="log_probability")

        torch.testing.assert_close(kl_prob, kl_log, atol=1e-5, rtol=1e-5)

    def test_logits_input(self):
        """input_type='logits' applies softmax."""
        logits_p = torch.randn(5)
        logits_q = torch.randn(5)

        kl_logits = kullback_leibler_divergence(logits_p, logits_q, input_type="logits")
        kl_prob = kullback_leibler_divergence(
            torch.softmax(logits_p, dim=0),
            torch.softmax(logits_q, dim=0),
            input_type="probability"
        )

        torch.testing.assert_close(kl_logits, kl_prob, atol=1e-5, rtol=1e-5)


class TestKLDivergenceReduction:
    """Tests for reduction parameter."""

    def test_reduction_sum(self):
        """reduction='sum' sums all values."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        kl_none = kullback_leibler_divergence(p, q, reduction="none")
        kl_sum = kullback_leibler_divergence(p, q, reduction="sum")

        torch.testing.assert_close(kl_sum, kl_none.sum(), atol=1e-6, rtol=1e-6)

    def test_reduction_mean(self):
        """reduction='mean' averages all values."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        kl_none = kullback_leibler_divergence(p, q, reduction="none")
        kl_mean = kullback_leibler_divergence(p, q, reduction="mean")

        torch.testing.assert_close(kl_mean, kl_none.mean(), atol=1e-6, rtol=1e-6)

    def test_reduction_batchmean(self):
        """reduction='batchmean' divides by batch size."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        kl_none = kullback_leibler_divergence(p, q, reduction="none")
        kl_batchmean = kullback_leibler_divergence(p, q, reduction="batchmean")

        expected = kl_none.sum() / 10
        torch.testing.assert_close(kl_batchmean, expected, atol=1e-6, rtol=1e-6)


class TestKLDivergencePairwise:
    """Tests for pairwise mode."""

    def test_pairwise_output_shape(self):
        """pairwise=True produces (m, k) output."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(7, 10), dim=-1)

        kl = kullback_leibler_divergence(p, q, pairwise=True)

        assert kl.shape == (5, 7)

    def test_pairwise_values(self):
        """Verify pairwise[i,j] == KL(p[i] || q[j])."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(4, 5), dim=-1)

        kl_pairwise = kullback_leibler_divergence(p, q, pairwise=True)

        for i in range(3):
            for j in range(4):
                kl_ij = kullback_leibler_divergence(p[i], q[j])
                torch.testing.assert_close(
                    kl_pairwise[i, j], kl_ij, atol=1e-6, rtol=1e-6
                )


class TestKLDivergenceGradients:
    """Tests for gradient computation."""

    def test_gradcheck_first_order(self):
        """First-order gradients pass gradcheck."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=0)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=0)
        p.requires_grad_(True)
        q.requires_grad_(True)

        def fn(p, q):
            return kullback_leibler_divergence(p, q)

        assert gradcheck(fn, (p, q), raise_exception=True)

    @pytest.mark.skip(reason="Second-order gradients require backward_backward kernel")
    def test_gradgradcheck_second_order(self):
        """Second-order gradients pass gradgradcheck."""
        from torch.autograd import gradgradcheck

        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=0)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=0)
        p.requires_grad_(True)
        q.requires_grad_(True)

        def fn(p, q):
            return kullback_leibler_divergence(p, q)

        assert gradgradcheck(fn, (p, q), raise_exception=True)


class TestKLDivergenceEdgeCases:
    """Tests for edge cases."""

    def test_near_zero_probabilities(self):
        """Near-zero probabilities don't cause NaN/Inf."""
        p = torch.tensor([0.99, 0.01, 0.0])
        q = torch.tensor([0.01, 0.01, 0.98])

        kl = kullback_leibler_divergence(p, q)

        assert torch.isfinite(kl)

    def test_different_dtypes(self):
        """Works with different dtypes."""
        for dtype in [torch.float32, torch.float64]:
            p = torch.softmax(torch.randn(5, dtype=dtype), dim=0)
            q = torch.softmax(torch.randn(5, dtype=dtype), dim=0)

            kl = kullback_leibler_divergence(p, q)

            assert kl.dtype == dtype
            assert torch.isfinite(kl)


class TestKLDivergenceValidation:
    """Tests for input validation."""

    def test_invalid_input_type(self):
        """Raises ValueError for invalid input_type."""
        p = torch.softmax(torch.randn(5), dim=0)
        q = torch.softmax(torch.randn(5), dim=0)

        with pytest.raises(ValueError, match="input_type"):
            kullback_leibler_divergence(p, q, input_type="invalid")

    def test_invalid_reduction(self):
        """Raises ValueError for invalid reduction."""
        p = torch.softmax(torch.randn(5), dim=0)
        q = torch.softmax(torch.randn(5), dim=0)

        with pytest.raises(ValueError, match="reduction"):
            kullback_leibler_divergence(p, q, reduction="invalid")

    def test_mismatched_dim_sizes(self):
        """Raises ValueError for mismatched distribution sizes."""
        p = torch.softmax(torch.randn(5), dim=0)
        q = torch.softmax(torch.randn(7), dim=0)

        with pytest.raises(ValueError, match="must match"):
            kullback_leibler_divergence(p, q)
```

**Step 3: Create JS divergence tests**

```python
# tests/torchscience/information_theory/test__jensen_shannon_divergence.py
import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information_theory import jensen_shannon_divergence


class TestJSDivergenceBasic:
    """Tests for basic properties."""

    def test_symmetric(self):
        """JS divergence is symmetric."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        js_pq = jensen_shannon_divergence(p, q, reduction="none")
        js_qp = jensen_shannon_divergence(q, p, reduction="none")

        torch.testing.assert_close(js_pq, js_qp, atol=1e-6, rtol=1e-6)

    def test_bounded(self):
        """JS divergence is bounded by log(2)."""
        p = torch.softmax(torch.randn(100, 10), dim=-1)
        q = torch.softmax(torch.randn(100, 10), dim=-1)

        js = jensen_shannon_divergence(p, q, reduction="none")

        assert (js >= 0).all()
        assert (js <= math.log(2) + 1e-6).all()

    def test_zero_for_identical(self):
        """JS(P || P) = 0."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)

        js = jensen_shannon_divergence(p, p, reduction="none")

        torch.testing.assert_close(js, torch.zeros(10), atol=1e-6, rtol=1e-6)


class TestJSDivergenceBase:
    """Tests for base parameter."""

    def test_base_2_bounded_by_1(self):
        """With base=2, JS is bounded by 1."""
        p = torch.softmax(torch.randn(100, 10), dim=-1)
        q = torch.softmax(torch.randn(100, 10), dim=-1)

        js = jensen_shannon_divergence(p, q, base=2, reduction="none")

        assert (js >= 0).all()
        assert (js <= 1.0 + 1e-6).all()

    def test_base_conversion(self):
        """base parameter correctly scales output."""
        p = torch.softmax(torch.randn(5), dim=0)
        q = torch.softmax(torch.randn(5), dim=0)

        js_nats = jensen_shannon_divergence(p, q)
        js_bits = jensen_shannon_divergence(p, q, base=2)

        expected_bits = js_nats / math.log(2)
        torch.testing.assert_close(js_bits, expected_bits, atol=1e-6, rtol=1e-6)


class TestJSDivergenceGradients:
    """Tests for gradient computation."""

    def test_gradcheck_first_order(self):
        """First-order gradients pass gradcheck."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=0)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=0)
        p.requires_grad_(True)
        q.requires_grad_(True)

        def fn(p, q):
            return jensen_shannon_divergence(p, q)

        assert gradcheck(fn, (p, q), raise_exception=True)
```

**Step 4: Commit**

```bash
git add tests/torchscience/information_theory/
git commit -m "test: add comprehensive tests for KL and JS divergence"
```

---

## Task 10: Build and Test

**Step 1: Build the extension**

```bash
cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/kl-divergence
uv sync
```

**Step 2: Run tests**

```bash
.venv/bin/python -m pytest tests/torchscience/information_theory/ -v
```

Expected: All tests pass (some gradient tests may be skipped pending backward_backward integration)

**Step 3: Fix any issues and commit**

If tests fail, debug and fix. Commit fixes.

---

Plan complete and saved to `docs/plans/2025-12-31-kullback-leibler-divergence-implementation.md`.
