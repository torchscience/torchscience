# Cook-Torrance BRDF Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `torchscience.graphics.shading.cook_torrance` as a differentiable Cook-Torrance specular BRDF using GGX microfacet distribution.

**Architecture:** Pure PyTorch implementation with C++ backend following the minkowski_distance pattern. Forward computes specular BRDF value using GGX distribution, Schlick-GGX geometry, and Schlick Fresnel. Backward computes gradients through the BRDF formula. Supports broadcast semantics for all inputs.

**Tech Stack:** C++17, PyTorch C++ API (ATen), torch.autograd, pybind11

### Design Decisions (from Brainstorming)

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Coordinate Space** | World space | Industry standard; caller handles tangent-space transforms |
| **Distribution (D)** | GGX/Trowbridge-Reitz | Universal standard (UE4, Unity, Mitsuba3) |
| **Geometry (G)** | Schlick-GGX with Smith | UE4 approach with Disney remapping |
| **Fresnel (F)** | Schlick approximation | Fast, accurate, universal |
| **Anisotropy** | Isotropic only (v1) | Simpler initial version; can extend later |
| **Return value** | Specular BRDF only | Clean separation; diffuse handled separately |
| **Batching** | Broadcast-compatible | Standard PyTorch semantics |

## API Specification

```python
def cook_torrance(
    normal: Tensor,           # (..., 3) - Surface normal (normalized)
    view: Tensor,             # (..., 3) - View direction (normalized, toward camera)
    light: Tensor,            # (..., 3) - Light direction (normalized, toward light)
    *,
    roughness: Tensor | float,  # (...) or scalar - Surface roughness [0, 1]
    f0: Tensor | float = 0.04,  # (...) or (..., 3) or scalar - Fresnel reflectance at normal incidence
) -> Tensor:
    """
    Compute Cook-Torrance specular BRDF.

    Returns the specular microfacet BRDF value for physically-based rendering.
    Uses GGX distribution, Schlick-GGX geometry, and Schlick Fresnel.

    Returns
    -------
    Tensor, shape (...) or (..., 3)
        Specular BRDF value. Shape matches broadcast of inputs.
        If f0 is RGB (..., 3), returns RGB; otherwise scalar per sample.
    """
```

---

## Mathematical Background

### Cook-Torrance Specular BRDF

The specular microfacet BRDF is:

$$f_r = \frac{D \cdot F \cdot G}{4(n \cdot l)(n \cdot v)}$$

Where:
- $D$ = Normal Distribution Function (GGX/Trowbridge-Reitz)
- $F$ = Fresnel term (Schlick approximation)
- $G$ = Geometry/masking-shadowing term (Schlick-GGX with Smith)
- $n$ = surface normal
- $l$ = light direction (toward light)
- $v$ = view direction (toward camera)
- $h$ = halfway vector = normalize(l + v)

### GGX/Trowbridge-Reitz Distribution (D)

$$D(h) = \frac{\alpha^2}{\pi((n \cdot h)^2(\alpha^2 - 1) + 1)^2}$$

Where $\alpha = \text{roughness}^2$ (Disney's reparameterization)

### Schlick-GGX Geometry (G)

$$G(l, v) = G_1(l) \cdot G_1(v)$$

$$G_1(x) = \frac{n \cdot x}{(n \cdot x)(1 - k) + k}$$

$$k = \frac{(\text{roughness} + 1)^2}{8}$$ (for direct lighting)

### Schlick Fresnel (F)

$$F(h, v) = F_0 + (1 - F_0)(1 - h \cdot v)^5$$

Where $F_0$ is the base reflectivity at normal incidence (typically 0.04 for dielectrics)

---

## Task 1: Create graphics.shading module Python structure

**Files:**
- Create: `src/torchscience/graphics/__init__.py`
- Create: `src/torchscience/graphics/shading/__init__.py`
- Create: `src/torchscience/graphics/shading/_cook_torrance.py`
- Modify: `src/torchscience/__init__.py`

**Step 1: Create the graphics module __init__.py**

```python
# src/torchscience/graphics/__init__.py
from . import shading

__all__ = [
    "shading",
]
```

**Step 2: Create the shading submodule __init__.py**

```python
# src/torchscience/graphics/shading/__init__.py
from ._cook_torrance import cook_torrance

__all__ = [
    "cook_torrance",
]
```

**Step 3: Create the Python API**

```python
# src/torchscience/graphics/shading/_cook_torrance.py
"""Cook-Torrance BRDF implementation."""

from typing import Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def cook_torrance(
    normal: Tensor,
    view: Tensor,
    light: Tensor,
    *,
    roughness: Union[Tensor, float],
    f0: Union[Tensor, float] = 0.04,
) -> Tensor:
    r"""Compute Cook-Torrance specular BRDF.

    Evaluates the Cook-Torrance microfacet specular BRDF using GGX distribution,
    Schlick-GGX geometry, and Schlick Fresnel approximation.

    Mathematical Definition
    -----------------------
    The specular BRDF is:

    .. math::
        f_r = \frac{D \cdot F \cdot G}{4(n \cdot l)(n \cdot v)}

    Where:

    - :math:`D`: GGX/Trowbridge-Reitz normal distribution
    - :math:`F`: Schlick Fresnel approximation
    - :math:`G`: Schlick-GGX geometry with Smith masking-shadowing

    Parameters
    ----------
    normal : Tensor, shape (..., 3)
        Surface normal vectors. Must be normalized.
    view : Tensor, shape (..., 3)
        View direction vectors (toward camera). Must be normalized.
    light : Tensor, shape (..., 3)
        Light direction vectors (toward light). Must be normalized.
    roughness : Tensor or float, shape (...) or scalar
        Surface roughness in range [0, 1]. Values are clamped to [0.001, 1.0]
        to avoid singularities.
    f0 : Tensor or float, shape (...) or (..., 3) or scalar, default=0.04
        Fresnel reflectance at normal incidence. Use 0.04 for common
        dielectrics (plastic, glass). For metals, use the material's
        RGB reflectance values.

    Returns
    -------
    Tensor
        Specular BRDF values. Shape depends on inputs:

        - If f0 is scalar or shape (...): returns shape (...)
        - If f0 is shape (..., 3): returns shape (..., 3) for RGB

    Examples
    --------
    Basic dielectric surface (plastic-like):

    >>> normal = torch.tensor([[0.0, 1.0, 0.0]])  # Up
    >>> view = torch.tensor([[0.0, 0.707, 0.707]])  # 45 degrees
    >>> light = torch.tensor([[0.0, 0.707, -0.707]])  # 45 degrees opposite
    >>> torchscience.graphics.shading.cook_torrance(
    ...     normal, view, light, roughness=0.5
    ... )
    tensor([...])

    Metallic surface with RGB reflectance:

    >>> f0_gold = torch.tensor([1.0, 0.71, 0.29])  # Gold reflectance
    >>> torchscience.graphics.shading.cook_torrance(
    ...     normal, view, light, roughness=0.3, f0=f0_gold
    ... )
    tensor([[...]])

    Notes
    -----
    - All direction vectors (normal, view, light) must be normalized.
    - The function returns 0 when n·l <= 0 or n·v <= 0 (back-facing).
    - Roughness is internally clamped to [0.001, 1.0] for numerical stability.
    - This implementation follows Unreal Engine 4's approach from
      "Real Shading in Unreal Engine 4" (SIGGRAPH 2013).

    References
    ----------
    .. [1] B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013.
           https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    .. [2] LearnOpenGL, "PBR Theory",
           https://learnopengl.com/PBR/Theory
    """
    # Input validation
    if normal.shape[-1] != 3:
        raise ValueError(f"normal must have last dimension 3, got {normal.shape[-1]}")
    if view.shape[-1] != 3:
        raise ValueError(f"view must have last dimension 3, got {view.shape[-1]}")
    if light.shape[-1] != 3:
        raise ValueError(f"light must have last dimension 3, got {light.shape[-1]}")

    # Convert scalars to tensors
    if not isinstance(roughness, Tensor):
        roughness = torch.tensor(roughness, dtype=normal.dtype, device=normal.device)
    if not isinstance(f0, Tensor):
        f0 = torch.tensor(f0, dtype=normal.dtype, device=normal.device)

    # Ensure tensors are on the same device and have compatible dtypes
    target_dtype = torch.promote_types(normal.dtype, view.dtype)
    target_dtype = torch.promote_types(target_dtype, light.dtype)
    target_device = normal.device

    if normal.dtype != target_dtype:
        normal = normal.to(target_dtype)
    if view.dtype != target_dtype:
        view = view.to(target_dtype)
    if light.dtype != target_dtype:
        light = light.to(target_dtype)
    if roughness.dtype != target_dtype:
        roughness = roughness.to(target_dtype)
    if f0.dtype != target_dtype:
        f0 = f0.to(target_dtype)

    if roughness.device != target_device:
        roughness = roughness.to(target_device)
    if f0.device != target_device:
        f0 = f0.to(target_device)

    return torch.ops.torchscience.cook_torrance(normal, view, light, roughness, f0)
```

**Step 4: Update main __init__.py to include graphics module**

Add `graphics` to the imports in `src/torchscience/__init__.py`:

```python
from . import (
    _csrc,
    distance,
    graphics,
    optimization,
    root_finding,
    signal_processing,
    statistics,
)

__all__ = [
    "_csrc",
    "distance",
    "graphics",
    "optimization",
    "root_finding",
    "signal_processing",
    "statistics",
]
```

**Step 5: Commit**

```bash
git add src/torchscience/graphics/ src/torchscience/__init__.py
git commit -m "feat(graphics): add module structure and Python API for cook_torrance"
```

---

## Task 2: Create impl header with forward math

**Files:**
- Create: `src/torchscience/csrc/impl/graphics/shading/cook_torrance.h`

**Step 1: Create forward implementation**

```cpp
// src/torchscience/csrc/impl/graphics/shading/cook_torrance.h
#pragma once

/*
 * Cook-Torrance BRDF Implementation
 *
 * MATHEMATICAL DEFINITION:
 * ========================
 * The Cook-Torrance specular BRDF is:
 *
 *   f_r = D * F * G / (4 * (n·l) * (n·v))
 *
 * Where:
 *   D = GGX/Trowbridge-Reitz normal distribution
 *   F = Schlick Fresnel approximation
 *   G = Schlick-GGX geometry with Smith masking-shadowing
 *
 * COMPONENT FORMULAS:
 * ===================
 *
 * GGX Distribution (D):
 *   D(h) = α² / (π * ((n·h)² * (α² - 1) + 1)²)
 *   where α = roughness²
 *
 * Schlick-GGX Geometry (G):
 *   G(l, v) = G₁(l) * G₁(v)
 *   G₁(x) = (n·x) / ((n·x)(1 - k) + k)
 *   k = (roughness + 1)² / 8  (for direct lighting)
 *
 * Schlick Fresnel (F):
 *   F(h, v) = F₀ + (1 - F₀) * (1 - h·v)⁵
 */

#include <c10/macros/Macros.h>
#include <cmath>

namespace torchscience::impl::graphics::shading {

// Minimum roughness to avoid division by zero
template <typename T>
constexpr T MIN_ROUGHNESS = T(0.001);

// Small epsilon for dot product clamping
template <typename T>
constexpr T DOT_EPSILON = T(1e-7);

/**
 * Compute GGX/Trowbridge-Reitz normal distribution function.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T ggx_distribution(T n_dot_h, T alpha_squared) {
    T n_dot_h_sq = n_dot_h * n_dot_h;
    T denom = n_dot_h_sq * (alpha_squared - T(1)) + T(1);
    denom = denom * denom;
    constexpr T PI = T(3.14159265358979323846);
    return alpha_squared / (PI * denom);
}

/**
 * Compute Schlick-GGX geometry sub-term.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T schlick_ggx_g1(T n_dot_x, T k) {
    return n_dot_x / (n_dot_x * (T(1) - k) + k);
}

/**
 * Compute Schlick-GGX geometry term (Smith masking-shadowing).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T schlick_ggx_geometry(T n_dot_v, T n_dot_l, T roughness) {
    T r_plus_1 = roughness + T(1);
    T k = (r_plus_1 * r_plus_1) / T(8);
    return schlick_ggx_g1(n_dot_v, k) * schlick_ggx_g1(n_dot_l, k);
}

/**
 * Compute Schlick Fresnel approximation.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T schlick_fresnel(T h_dot_v, T f0) {
    T one_minus_cos = T(1) - h_dot_v;
    T pow5 = one_minus_cos * one_minus_cos;
    pow5 = pow5 * pow5 * one_minus_cos;
    return f0 + (T(1) - f0) * pow5;
}

/**
 * Compute dot product of two 3D vectors.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T dot3(const T* a, const T* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Compute Cook-Torrance specular BRDF for a single sample.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T cook_torrance_scalar(
    const T* normal,
    const T* view,
    const T* light,
    T roughness,
    T f0
) {
    // Clamp roughness to avoid singularities
    roughness = std::max(roughness, MIN_ROUGHNESS<T>);

    // Compute dot products
    T n_dot_l = dot3(normal, light);
    T n_dot_v = dot3(normal, view);

    // Early out for back-facing geometry
    if (n_dot_l <= T(0) || n_dot_v <= T(0)) {
        return T(0);
    }

    n_dot_l = std::max(n_dot_l, DOT_EPSILON<T>);
    n_dot_v = std::max(n_dot_v, DOT_EPSILON<T>);

    // Compute halfway vector: h = normalize(l + v)
    T h[3] = { light[0] + view[0], light[1] + view[1], light[2] + view[2] };
    T h_len = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);

    if (h_len < DOT_EPSILON<T>) {
        return T(0);
    }

    T inv_h_len = T(1) / h_len;
    T h_normalized[3] = { h[0] * inv_h_len, h[1] * inv_h_len, h[2] * inv_h_len };

    T n_dot_h = std::max(dot3(normal, h_normalized), T(0));
    T h_dot_v = std::max(dot3(h_normalized, view), T(0));

    // Compute BRDF components
    T alpha = roughness * roughness;
    T alpha_squared = alpha * alpha;

    T D = ggx_distribution(n_dot_h, alpha_squared);
    T G = schlick_ggx_geometry(n_dot_v, n_dot_l, roughness);
    T F = schlick_fresnel(h_dot_v, f0);

    // Cook-Torrance specular BRDF
    T denominator = T(4) * n_dot_l * n_dot_v;
    return (D * G * F) / denominator;
}

}  // namespace torchscience::impl::graphics::shading
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/impl/graphics/shading/
git commit -m "feat(impl): add Cook-Torrance forward implementation"
```

---

## Task 3: Create impl header with backward math

**Files:**
- Create: `src/torchscience/csrc/impl/graphics/shading/cook_torrance_backward.h`

**Step 1: Create backward implementation**

```cpp
// src/torchscience/csrc/impl/graphics/shading/cook_torrance_backward.h
#pragma once

#include <c10/macros/Macros.h>
#include <cmath>
#include "cook_torrance.h"

namespace torchscience::impl::graphics::shading {

/**
 * Compute gradients for Cook-Torrance BRDF.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void cook_torrance_backward_scalar(
    T grad_out,
    const T* normal,
    const T* view,
    const T* light,
    T roughness,
    T f0,
    T* grad_normal,
    T* grad_view,
    T* grad_light,
    T* grad_roughness,
    T* grad_f0
) {
    // Initialize gradients to zero
    for (int i = 0; i < 3; ++i) {
        grad_normal[i] = T(0);
        grad_view[i] = T(0);
        grad_light[i] = T(0);
    }
    *grad_roughness = T(0);
    *grad_f0 = T(0);

    roughness = std::max(roughness, MIN_ROUGHNESS<T>);

    T n_dot_l = dot3(normal, light);
    T n_dot_v = dot3(normal, view);

    if (n_dot_l <= T(0) || n_dot_v <= T(0)) {
        return;
    }

    T n_dot_l_clamped = std::max(n_dot_l, DOT_EPSILON<T>);
    T n_dot_v_clamped = std::max(n_dot_v, DOT_EPSILON<T>);

    T h[3] = { light[0] + view[0], light[1] + view[1], light[2] + view[2] };
    T h_len = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);

    if (h_len < DOT_EPSILON<T>) {
        return;
    }

    T inv_h_len = T(1) / h_len;
    T h_normalized[3] = { h[0] * inv_h_len, h[1] * inv_h_len, h[2] * inv_h_len };

    T n_dot_h = std::max(dot3(normal, h_normalized), DOT_EPSILON<T>);
    T h_dot_v = std::max(dot3(h_normalized, view), DOT_EPSILON<T>);

    T alpha = roughness * roughness;
    T alpha_squared = alpha * alpha;

    // D term
    T n_dot_h_sq = n_dot_h * n_dot_h;
    T denom_d = n_dot_h_sq * (alpha_squared - T(1)) + T(1);
    T denom_d_sq = denom_d * denom_d;
    constexpr T PI = T(3.14159265358979323846);
    T D = alpha_squared / (PI * denom_d_sq);

    // G term
    T r_plus_1 = roughness + T(1);
    T k = (r_plus_1 * r_plus_1) / T(8);
    T g1_v_denom = n_dot_v_clamped * (T(1) - k) + k;
    T g1_l_denom = n_dot_l_clamped * (T(1) - k) + k;
    T G1_v = n_dot_v_clamped / g1_v_denom;
    T G1_l = n_dot_l_clamped / g1_l_denom;
    T G = G1_v * G1_l;

    // F term
    T one_minus_hdv = T(1) - h_dot_v;
    T pow4 = one_minus_hdv * one_minus_hdv;
    pow4 = pow4 * pow4;
    T pow5 = pow4 * one_minus_hdv;
    T F = f0 + (T(1) - f0) * pow5;

    T denom = T(4) * n_dot_l_clamped * n_dot_v_clamped;
    T brdf = (D * G * F) / denom;

    // Gradient w.r.t. F₀
    T dF_df0 = T(1) - pow5;
    T d_brdf_dF = D * G / denom;
    *grad_f0 = grad_out * d_brdf_dF * dF_df0;

    // Gradient w.r.t. roughness
    T d_denom_d_dalpha_sq = n_dot_h_sq;
    T dD_dalpha_sq = (denom_d - T(2) * alpha_squared * d_denom_d_dalpha_sq) / (PI * denom_d_sq * denom_d);
    T dalpha_sq_droughness = T(4) * alpha * roughness;

    T dk_droughness = (roughness + T(1)) / T(4);
    T dG1_v_dk = n_dot_v_clamped * (n_dot_v_clamped - T(1)) / (g1_v_denom * g1_v_denom);
    T dG1_l_dk = n_dot_l_clamped * (n_dot_l_clamped - T(1)) / (g1_l_denom * g1_l_denom);
    T dG_dk = dG1_v_dk * G1_l + G1_v * dG1_l_dk;
    T dG_droughness = dG_dk * dk_droughness;

    T d_brdf_dD = G * F / denom;
    T d_brdf_dG = D * F / denom;

    *grad_roughness = grad_out * (d_brdf_dD * dD_dalpha_sq * dalpha_sq_droughness + d_brdf_dG * dG_droughness);

    // Vector gradients
    T dD_dndoth = -T(4) * alpha_squared * n_dot_h * (alpha_squared - T(1)) / (PI * denom_d_sq * denom_d);
    T d_brdf_dndoth = grad_out * d_brdf_dD * dD_dndoth;

    T dF_dhdotv = -T(5) * (T(1) - f0) * pow4;
    T d_brdf_dhdotv = grad_out * d_brdf_dF * dF_dhdotv;

    T dG1_v_dndotv = k / (g1_v_denom * g1_v_denom);
    T dG_dndotv = dG1_v_dndotv * G1_l;
    T d_brdf_dndotv = grad_out * (d_brdf_dG * dG_dndotv - brdf / n_dot_v_clamped);

    T dG1_l_dndotl = k / (g1_l_denom * g1_l_denom);
    T dG_dndotl = G1_v * dG1_l_dndotl;
    T d_brdf_dndotl = grad_out * (d_brdf_dG * dG_dndotl - brdf / n_dot_l_clamped);

    // Propagate to vectors
    for (int i = 0; i < 3; ++i) {
        grad_normal[i] = d_brdf_dndotl * light[i] + d_brdf_dndotv * view[i] + d_brdf_dndoth * h_normalized[i];
    }

    T dndoth_dv[3], dndoth_dl[3];
    for (int i = 0; i < 3; ++i) {
        dndoth_dv[i] = (normal[i] - n_dot_h * h_normalized[i]) * inv_h_len;
        dndoth_dl[i] = dndoth_dv[i];
    }

    T dhdotv_dv[3], dhdotv_dl[3];
    for (int i = 0; i < 3; ++i) {
        dhdotv_dv[i] = h_normalized[i] + (view[i] - h_dot_v * h_normalized[i]) * inv_h_len;
        dhdotv_dl[i] = (view[i] - h_dot_v * h_normalized[i]) * inv_h_len;
    }

    for (int i = 0; i < 3; ++i) {
        grad_view[i] = d_brdf_dndotv * normal[i] + d_brdf_dndoth * dndoth_dv[i] + d_brdf_dhdotv * dhdotv_dv[i];
        grad_light[i] = d_brdf_dndotl * normal[i] + d_brdf_dndoth * dndoth_dl[i] + d_brdf_dhdotv * dhdotv_dl[i];
    }
}

}  // namespace torchscience::impl::graphics::shading
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/impl/graphics/shading/cook_torrance_backward.h
git commit -m "feat(impl): add Cook-Torrance backward implementation"
```

---

## Task 4: Create CPU kernel

**Files:**
- Create: `src/torchscience/csrc/cpu/graphics/shading/cook_torrance.h`

**Step 1: Create CPU implementation with forward and backward**

The CPU kernel should:
- Use `AT_DISPATCH_FLOATING_TYPES_AND2` for dtype dispatch
- Use `at::parallel_for` for parallelization
- Handle broadcasting across batch dimensions
- Include both forward and backward implementations
- Register with `TORCH_LIBRARY_IMPL(torchscience, CPU, module)`

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/graphics/shading/
git commit -m "feat(cpu): add Cook-Torrance CPU kernel implementation"
```

---

## Task 5: Create autograd wrapper

**Files:**
- Create: `src/torchscience/csrc/autograd/graphics/shading/cook_torrance.h`

**Step 1: Create autograd Function class**

The autograd wrapper should:
- Create `CookTorrance` class inheriting from `torch::autograd::Function`
- Implement `forward()` saving tensors for backward
- Implement `backward()` calling the backward dispatcher
- Register with `TORCH_LIBRARY_IMPL(torchscience, Autograd, module)`

**Step 2: Commit**

```bash
git add src/torchscience/csrc/autograd/graphics/shading/
git commit -m "feat(autograd): add Cook-Torrance autograd wrapper"
```

---

## Task 6: Create meta tensor implementation

**Files:**
- Create: `src/torchscience/csrc/meta/graphics/shading/cook_torrance.h`

**Step 1: Create meta implementation for shape inference**

The meta implementation should:
- Return output shape based on input shapes
- Handle RGB vs scalar f0 cases
- Register with `TORCH_LIBRARY_IMPL(torchscience, Meta, m)`

**Step 2: Commit**

```bash
git add src/torchscience/csrc/meta/graphics/shading/
git commit -m "feat(meta): add Cook-Torrance meta tensor implementation"
```

---

## Task 7: Register operators and update main cpp

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp`

**Step 1: Add operator schema definitions**

Add to the `TORCH_LIBRARY(torchscience, module)` block:

```cpp
// `torchscience.graphics.shading`
module.def("cook_torrance(Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> Tensor");
module.def("cook_torrance_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
```

**Step 2: Add includes for the new headers**

```cpp
#include "cpu/graphics/shading/cook_torrance.h"
#include "autograd/graphics/shading/cook_torrance.h"
#include "meta/graphics/shading/cook_torrance.h"
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat: register cook_torrance operators in TORCH_LIBRARY"
```

---

## Task 8: Write comprehensive tests

**Files:**
- Create: `tests/torchscience/graphics/__init__.py`
- Create: `tests/torchscience/graphics/shading/__init__.py`
- Create: `tests/torchscience/graphics/shading/test__cook_torrance.py`

**Step 1: Create test directory structure**

**Step 2: Write comprehensive tests covering:**
- Basic functionality (output shapes, non-negative values)
- Correctness (back-facing returns 0, mirror reflection peak, roughness effects)
- Validation (dimension checks)
- Gradients (gradcheck, finite gradients)
- Dtype support (float32, float64)
- Device support (CPU, CUDA if available)

**Step 3: Commit**

```bash
git add tests/torchscience/graphics/
git commit -m "test: add comprehensive tests for cook_torrance"
```

---

## Task 9: Build and verify

**Step 1: Build the project**

```bash
uv run pip install -e .
```

**Step 2: Run all cook_torrance tests**

```bash
uv run pytest tests/torchscience/graphics/shading/test__cook_torrance.py -v
```

**Step 3: Verify import works**

```bash
uv run python -c "from torchscience.graphics.shading import cook_torrance; print(cook_torrance)"
```

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address any build or test issues"
```

---

## Summary

This plan implements `torchscience.graphics.shading.cook_torrance` as a differentiable Cook-Torrance specular BRDF:

1. **Python API** - User-facing function with validation and documentation
2. **impl headers** - Device-agnostic math for forward (GGX, Schlick-GGX, Schlick Fresnel) and backward
3. **CPU kernel** - Parallelized BRDF computation with broadcasting
4. **Autograd wrapper** - Gradient support through torch.autograd.Function
5. **Meta implementation** - Shape inference for torch.compile
6. **Operator registration** - TORCH_LIBRARY schema definitions
7. **Tests** - Comprehensive correctness, gradient, and device tests
8. **Build verification** - Ensure everything compiles and runs

**Key features:**
- GGX/Trowbridge-Reitz normal distribution (industry standard)
- Schlick-GGX geometry with Smith masking-shadowing
- Schlick Fresnel approximation
- Supports scalar or RGB F0 for metals
- Full autograd support for inverse rendering applications
- Follows Unreal Engine 4's implementation approach

**References:**
- [Real Shading in Unreal Engine 4 - Brian Karis (SIGGRAPH 2013)](https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf)
- [LearnOpenGL - PBR Theory](https://learnopengl.com/PBR/Theory)
- [Mitsuba 3 BSDF Documentation](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html)
