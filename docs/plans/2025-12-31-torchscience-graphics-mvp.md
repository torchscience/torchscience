# torchscience.graphics MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement MVP operators for `torchscience.graphics` covering 6 algorithm areas with 9 new operators.

**Architecture:** Each operator follows the established pattern: Python API → C++ schema registration → CPU kernel + Meta implementation + Autograd wrapper. Operators take raw tensors (loose coupling with geometry module).

**Tech Stack:** PyTorch C++ extensions, TORCH_LIBRARY for schema, AT_DISPATCH_FLOATING_TYPES for kernels.

---

## Summary

| Submodule | Operator | Status |
|-----------|----------|--------|
| `color` | `srgb_to_hsv`, `hsv_to_srgb` | ✓ Done |
| `shading` | `cook_torrance` | ✓ Done |
| `shading` | `lambertian` | To implement |
| `shading` | `phong` | To implement |
| `shading` | `blinn_phong` | To implement |
| `light_sources` | `point_light` | To implement |
| `light_sources` | `spotlight` | To implement |
| `light_sources` | `directional_light` | To implement |
| `tone_mapping` | `reinhard` | To implement |
| `texture_mapping` | `cube_mapping` | To implement |
| `projection` | `perspective_projection` | To implement |

---

## Task 1: Add `lambertian` shading operator

**Files:**
- Create: `src/torchscience/csrc/cpu/graphics/shading/lambertian.h`
- Create: `src/torchscience/csrc/meta/graphics/shading/lambertian.h`
- Create: `src/torchscience/csrc/autograd/graphics/shading/lambertian.h`
- Modify: `src/torchscience/csrc/torchscience.cpp` (add includes and schema)
- Create: `src/torchscience/graphics/shading/_lambertian.py`
- Modify: `src/torchscience/graphics/shading/__init__.py`
- Create: `tests/torchscience/graphics/shading/test__lambertian.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/graphics/shading/test__lambertian.py
"""Tests for Lambertian reflectance."""

import math
import pytest
import torch
from torch.autograd import gradcheck


class TestLambertianBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_single_sample(self):
        """Output shape matches batch dimensions for single sample."""
        from torchscience.graphics.shading import lambertian

        normal = torch.tensor([[0.0, 1.0, 0.0]])
        light = torch.tensor([[0.0, 0.707, 0.707]])

        result = lambertian(normal, light)

        assert result.shape == (1,)

    def test_output_shape_batch(self):
        """Output shape matches batch dimensions for batched input."""
        from torchscience.graphics.shading import lambertian

        normal = torch.randn(10, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        light = torch.randn(10, 3)
        light = light / light.norm(dim=-1, keepdim=True)

        result = lambertian(normal, light)

        assert result.shape == (10,)

    def test_reflectance_non_negative(self):
        """Reflectance values are always non-negative."""
        from torchscience.graphics.shading import lambertian

        normal = torch.randn(100, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        light = torch.randn(100, 3)
        light = light / light.norm(dim=-1, keepdim=True)

        result = lambertian(normal, light)

        assert (result >= 0).all()


class TestLambertianCorrectness:
    """Tests for numerical correctness."""

    def test_back_facing_returns_zero(self):
        """Reflectance returns 0 when surface is back-facing."""
        from torchscience.graphics.shading import lambertian

        normal = torch.tensor([[0.0, 1.0, 0.0]])
        light = torch.tensor([[0.0, -0.5, 0.866]])  # Below horizon

        result = lambertian(normal, light)

        assert result.item() == 0.0

    def test_normal_incidence_value(self):
        """Reflectance at normal incidence equals 1/pi."""
        from torchscience.graphics.shading import lambertian

        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        light = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)

        result = lambertian(normal, light)

        # At normal incidence, n·l = 1, so result = 1/pi
        expected = 1.0 / math.pi
        torch.testing.assert_close(
            result,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_45_degree_incidence_value(self):
        """Reflectance at 45 degrees equals cos(45)/pi."""
        from torchscience.graphics.shading import lambertian

        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        light = torch.tensor([[0.0, 0.707106781, 0.707106781]], dtype=torch.float64)

        result = lambertian(normal, light)

        # n·l = cos(45) ≈ 0.707
        expected = 0.707106781 / math.pi
        torch.testing.assert_close(
            result,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-7,
        )


class TestLambertianGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Passes gradcheck for basic inputs."""
        from torchscience.graphics.shading import lambertian

        normal = torch.randn(3, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        normal = normal.detach().requires_grad_(True)

        light = torch.randn(3, 3, dtype=torch.float64)
        light = light / light.norm(dim=-1, keepdim=True)
        light = light.detach().requires_grad_(True)

        def func(n, l):
            return lambertian(n, l)

        assert gradcheck(func, (normal, light), raise_exception=True)

    def test_gradients_finite(self):
        """Gradients are finite for typical inputs."""
        from torchscience.graphics.shading import lambertian

        normal = torch.tensor(
            [[0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        light = torch.tensor(
            [[0.0, 0.707, 0.707]], dtype=torch.float64, requires_grad=True
        )

        result = lambertian(normal, light)
        result.sum().backward()

        assert normal.grad is not None and torch.isfinite(normal.grad).all()
        assert light.grad is not None and torch.isfinite(light.grad).all()


class TestLambertianValidation:
    """Tests for input validation."""

    def test_normal_wrong_dimension(self):
        """Raises error when normal last dimension != 3."""
        from torchscience.graphics.shading import lambertian

        normal = torch.randn(10, 2)
        light = torch.randn(10, 3)

        with pytest.raises(ValueError, match="normal must have last dimension 3"):
            lambertian(normal, light)

    def test_light_wrong_dimension(self):
        """Raises error when light last dimension != 3."""
        from torchscience.graphics.shading import lambertian

        normal = torch.randn(10, 3)
        light = torch.randn(10, 2)

        with pytest.raises(ValueError, match="light must have last dimension 3"):
            lambertian(normal, light)


class TestLambertianDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.shading import lambertian

        normal = torch.randn(5, 3, dtype=torch.float32)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        light = torch.randn(5, 3, dtype=torch.float32)
        light = light / light.norm(dim=-1, keepdim=True)

        result = lambertian(normal, light)

        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.graphics.shading import lambertian

        normal = torch.randn(5, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        light = torch.randn(5, 3, dtype=torch.float64)
        light = light / light.norm(dim=-1, keepdim=True)

        result = lambertian(normal, light)

        assert result.dtype == torch.float64
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/graphics/shading/test__lambertian.py -v`
Expected: FAIL with "cannot import name 'lambertian'"

**Step 3: Create CPU kernel**

```cpp
// src/torchscience/csrc/cpu/graphics/shading/lambertian.h
#pragma once

#include <cmath>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graphics::shading {

namespace {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T dot3(const T* a, const T* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T lambertian_scalar(const T* normal, const T* light) {
    const T PI = T(3.14159265358979323846);
    T n_dot_l = dot3(normal, light);
    return n_dot_l > T(0) ? n_dot_l / PI : T(0);
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void lambertian_backward_scalar(
    T grad_output,
    const T* normal,
    const T* light,
    T* grad_normal,
    T* grad_light
) {
    const T PI = T(3.14159265358979323846);
    T n_dot_l = dot3(normal, light);

    if (n_dot_l > T(0)) {
        T scale = grad_output / PI;
        for (int i = 0; i < 3; ++i) {
            grad_normal[i] = scale * light[i];
            grad_light[i] = scale * normal[i];
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            grad_normal[i] = T(0);
            grad_light[i] = T(0);
        }
    }
}

}  // anonymous namespace

inline at::Tensor lambertian(
    const at::Tensor& normal,
    const at::Tensor& light
) {
    TORCH_CHECK(normal.size(-1) == 3, "lambertian: normal must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "lambertian: light must have last dimension 3");

    auto normal_contig = normal.contiguous();
    auto light_contig = light.contiguous();

    // Compute output shape (batch dimensions)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < normal.dim() - 1; ++i) {
        output_shape.push_back(normal.size(i));
    }

    auto output = at::empty(output_shape, normal.options());
    int64_t num_elements = output.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        normal.scalar_type(), "lambertian_cpu", [&] {
            const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_contig.data_ptr<scalar_t>();
            scalar_t* output_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    output_ptr[i] = lambertian_scalar(
                        normal_ptr + i * 3,
                        light_ptr + i * 3
                    );
                }
            });
        }
    );

    return output;
}

inline std::tuple<at::Tensor, at::Tensor> lambertian_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& light
) {
    auto grad_output_contig = grad_output.contiguous();
    auto normal_contig = normal.contiguous();
    auto light_contig = light.contiguous();

    auto grad_normal = at::empty_like(normal);
    auto grad_light = at::empty_like(light);

    int64_t num_elements = grad_output.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        normal.scalar_type(), "lambertian_backward_cpu", [&] {
            const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
            const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_contig.data_ptr<scalar_t>();
            scalar_t* grad_normal_ptr = grad_normal.data_ptr<scalar_t>();
            scalar_t* grad_light_ptr = grad_light.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    lambertian_backward_scalar(
                        grad_output_ptr[i],
                        normal_ptr + i * 3,
                        light_ptr + i * 3,
                        grad_normal_ptr + i * 3,
                        grad_light_ptr + i * 3
                    );
                }
            });
        }
    );

    return std::make_tuple(grad_normal, grad_light);
}

}  // namespace torchscience::cpu::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("lambertian", &torchscience::cpu::graphics::shading::lambertian);
    m.impl("lambertian_backward", &torchscience::cpu::graphics::shading::lambertian_backward);
}
```

**Step 4: Create Meta implementation**

```cpp
// src/torchscience/csrc/meta/graphics/shading/lambertian.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::shading {

inline at::Tensor lambertian(
    const at::Tensor& normal,
    const at::Tensor& light
) {
    TORCH_CHECK(normal.size(-1) == 3, "lambertian: normal must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "lambertian: light must have last dimension 3");

    // Output shape is batch dimensions (exclude last dim of 3)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < normal.dim() - 1; ++i) {
        output_shape.push_back(normal.size(i));
    }

    return at::empty(output_shape, normal.options());
}

inline std::tuple<at::Tensor, at::Tensor> lambertian_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& light
) {
    return std::make_tuple(
        at::empty_like(normal),
        at::empty_like(light)
    );
}

}  // namespace torchscience::meta::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("lambertian", &torchscience::meta::graphics::shading::lambertian);
    m.impl("lambertian_backward", &torchscience::meta::graphics::shading::lambertian_backward);
}
```

**Step 5: Create Autograd wrapper**

```cpp
// src/torchscience/csrc/autograd/graphics/shading/lambertian.h
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::graphics::shading {

class Lambertian : public torch::autograd::Function<Lambertian> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& normal,
        const at::Tensor& light
    ) {
        bool normal_requires_grad = normal.requires_grad();
        bool light_requires_grad = light.requires_grad();

        ctx->saved_data["normal_requires_grad"] = normal_requires_grad;
        ctx->saved_data["light_requires_grad"] = light_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::lambertian", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(normal, light);

        ctx->save_for_backward({normal, light});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor normal = saved[0];
        at::Tensor light = saved[1];

        at::Tensor grad_output = grad_outputs[0];

        bool normal_requires_grad = ctx->saved_data["normal_requires_grad"].toBool();
        bool light_requires_grad = ctx->saved_data["light_requires_grad"].toBool();

        if (!normal_requires_grad && !light_requires_grad) {
            return {at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_normal, grad_light] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::lambertian_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output, normal, light);

        return {
            normal_requires_grad ? grad_normal : at::Tensor(),
            light_requires_grad ? grad_light : at::Tensor()
        };
    }
};

inline at::Tensor lambertian(
    const at::Tensor& normal,
    const at::Tensor& light
) {
    return Lambertian::apply(normal, light);
}

}  // namespace torchscience::autograd::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("lambertian", &torchscience::autograd::graphics::shading::lambertian);
}
```

**Step 6: Add includes and schema to torchscience.cpp**

Add these includes after the existing graphics includes:
```cpp
#include "cpu/graphics/shading/lambertian.h"
#include "autograd/graphics/shading/lambertian.h"
#include "meta/graphics/shading/lambertian.h"
```

Add these schema definitions in TORCH_LIBRARY after cook_torrance:
```cpp
module.def("lambertian(Tensor normal, Tensor light) -> Tensor");
module.def("lambertian_backward(Tensor grad_output, Tensor normal, Tensor light) -> (Tensor, Tensor)");
```

**Step 7: Create Python API**

```python
# src/torchscience/graphics/shading/_lambertian.py
"""Lambertian reflectance implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def lambertian(normal: Tensor, light: Tensor) -> Tensor:
    r"""Compute Lambertian diffuse reflectance.

    Evaluates the Lambertian BRDF which models ideal diffuse reflection.
    The reflected light is uniform in all directions.

    Mathematical Definition
    -----------------------
    The Lambertian BRDF is constant:

    .. math::
        f_r = \frac{1}{\pi}

    The reflected radiance is:

    .. math::
        L_r = \frac{\max(0, n \cdot l)}{\pi}

    Parameters
    ----------
    normal : Tensor, shape (..., 3)
        Surface normal vectors. Must be normalized.
    light : Tensor, shape (..., 3)
        Light direction vectors (toward light). Must be normalized.

    Returns
    -------
    Tensor, shape (...)
        Diffuse reflectance values. The output shape is the batch
        dimensions of the input (excluding the last dimension of 3).

    Examples
    --------
    Basic usage:

    >>> normal = torch.tensor([[0.0, 1.0, 0.0]])
    >>> light = torch.tensor([[0.0, 0.707, 0.707]])
    >>> torchscience.graphics.shading.lambertian(normal, light)
    tensor([0.2251])

    Batched computation:

    >>> normal = torch.randn(100, 3)
    >>> normal = normal / normal.norm(dim=-1, keepdim=True)
    >>> light = torch.randn(100, 3)
    >>> light = light / light.norm(dim=-1, keepdim=True)
    >>> result = torchscience.graphics.shading.lambertian(normal, light)
    >>> result.shape
    torch.Size([100])

    Notes
    -----
    - All direction vectors must be normalized.
    - Returns 0 when n·l <= 0 (back-facing geometry).
    - Division by π ensures energy conservation (integrates to 1 over hemisphere).
    - Gradients are computed analytically and support backpropagation.

    References
    ----------
    .. [1] Wikipedia, "Lambertian reflectance",
           https://en.wikipedia.org/wiki/Lambertian_reflectance
    """
    if normal.shape[-1] != 3:
        raise ValueError(
            f"normal must have last dimension 3, got {normal.shape[-1]}"
        )
    if light.shape[-1] != 3:
        raise ValueError(
            f"light must have last dimension 3, got {light.shape[-1]}"
        )

    return torch.ops.torchscience.lambertian(normal, light)
```

**Step 8: Update shading __init__.py**

```python
# src/torchscience/graphics/shading/__init__.py
from ._cook_torrance import cook_torrance
from ._lambertian import lambertian

__all__ = [
    "cook_torrance",
    "lambertian",
]
```

**Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/torchscience/graphics/shading/test__lambertian.py -v`
Expected: All tests PASS

**Step 10: Commit**

```bash
git add src/torchscience/csrc/cpu/graphics/shading/lambertian.h \
        src/torchscience/csrc/meta/graphics/shading/lambertian.h \
        src/torchscience/csrc/autograd/graphics/shading/lambertian.h \
        src/torchscience/csrc/torchscience.cpp \
        src/torchscience/graphics/shading/_lambertian.py \
        src/torchscience/graphics/shading/__init__.py \
        tests/torchscience/graphics/shading/test__lambertian.py
git commit -m "feat(shading): add lambertian reflectance operator"
```

---

## Task 2: Add `phong` shading operator

**Files:**
- Create: `src/torchscience/csrc/cpu/graphics/shading/phong.h`
- Create: `src/torchscience/csrc/meta/graphics/shading/phong.h`
- Create: `src/torchscience/csrc/autograd/graphics/shading/phong.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/graphics/shading/_phong.py`
- Modify: `src/torchscience/graphics/shading/__init__.py`
- Create: `tests/torchscience/graphics/shading/test__phong.py`

Follow the same pattern as Task 1. Key differences:

**Phong formula:**
```
R = 2(n·l)n - l  (reflection vector)
specular = max(0, R·v)^shininess
```

**Schema:**
```cpp
module.def("phong(Tensor normal, Tensor view, Tensor light, Tensor shininess) -> Tensor");
module.def("phong_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor shininess) -> (Tensor, Tensor, Tensor, Tensor)");
```

---

## Task 3: Add `blinn_phong` shading operator

**Files:**
- Create: `src/torchscience/csrc/cpu/graphics/shading/blinn_phong.h`
- Create: `src/torchscience/csrc/meta/graphics/shading/blinn_phong.h`
- Create: `src/torchscience/csrc/autograd/graphics/shading/blinn_phong.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/graphics/shading/_blinn_phong.py`
- Modify: `src/torchscience/graphics/shading/__init__.py`
- Create: `tests/torchscience/graphics/shading/test__blinn_phong.py`

**Blinn-Phong formula:**
```
H = normalize(L + V)  (half vector)
specular = max(0, n·H)^shininess
```

---

## Task 4: Add `light_sources` submodule with `point_light`

**Files:**
- Create: `src/torchscience/graphics/light_sources/__init__.py`
- Create: `src/torchscience/graphics/light_sources/_point_light.py`
- Create: `src/torchscience/csrc/cpu/graphics/light_sources/point_light.h`
- Create: `src/torchscience/csrc/meta/graphics/light_sources/point_light.h`
- Create: `src/torchscience/csrc/autograd/graphics/light_sources/point_light.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Modify: `src/torchscience/graphics/__init__.py`
- Create: `tests/torchscience/graphics/light_sources/__init__.py`
- Create: `tests/torchscience/graphics/light_sources/test__point_light.py`

**Point light formula:**
```
direction = normalize(light_position - surface_position)
distance = ||light_position - surface_position||
attenuation = 1 / (distance^2)
irradiance = intensity * attenuation
```

**Schema:**
```cpp
module.def("point_light(Tensor light_position, Tensor surface_position, Tensor intensity) -> (Tensor, Tensor)");
// Returns (irradiance, light_direction)
```

---

## Task 5: Add `spotlight` operator

**Files:**
- Create: `src/torchscience/csrc/cpu/graphics/light_sources/spotlight.h`
- Create: `src/torchscience/csrc/meta/graphics/light_sources/spotlight.h`
- Create: `src/torchscience/csrc/autograd/graphics/light_sources/spotlight.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/graphics/light_sources/_spotlight.py`
- Modify: `src/torchscience/graphics/light_sources/__init__.py`
- Create: `tests/torchscience/graphics/light_sources/test__spotlight.py`

**Spotlight formula:**
```
theta = acos(dot(-light_direction, spot_direction))
falloff = smoothstep(cos(outer_angle), cos(inner_angle), cos(theta))
irradiance = point_light_irradiance * falloff
```

---

## Task 6: Add `directional_light` operator

**Files:**
- Create: `src/torchscience/csrc/cpu/graphics/light_sources/directional_light.h`
- Create: `src/torchscience/csrc/meta/graphics/light_sources/directional_light.h`
- Create: `src/torchscience/csrc/autograd/graphics/light_sources/directional_light.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/graphics/light_sources/_directional_light.py`
- Modify: `src/torchscience/graphics/light_sources/__init__.py`
- Create: `tests/torchscience/graphics/light_sources/test__directional_light.py`

**Directional light formula:**
```
irradiance = intensity  (constant, no falloff)
light_direction = -direction  (toward the light)
```

---

## Task 7: Add `tone_mapping` submodule with `reinhard`

**Files:**
- Create: `src/torchscience/graphics/tone_mapping/__init__.py`
- Create: `src/torchscience/graphics/tone_mapping/_reinhard.py`
- Create: `src/torchscience/csrc/cpu/graphics/tone_mapping/reinhard.h`
- Create: `src/torchscience/csrc/meta/graphics/tone_mapping/reinhard.h`
- Create: `src/torchscience/csrc/autograd/graphics/tone_mapping/reinhard.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Modify: `src/torchscience/graphics/__init__.py`
- Create: `tests/torchscience/graphics/tone_mapping/__init__.py`
- Create: `tests/torchscience/graphics/tone_mapping/test__reinhard.py`

**Reinhard formula:**
```
Basic: L_out = L_in / (1 + L_in)
Extended: L_out = L_in * (1 + L_in/L_white^2) / (1 + L_in)
```

---

## Task 8: Add `texture_mapping` submodule with `cube_mapping`

**Files:**
- Create: `src/torchscience/graphics/texture_mapping/__init__.py`
- Create: `src/torchscience/graphics/texture_mapping/_cube_mapping.py`
- Create: `src/torchscience/csrc/cpu/graphics/texture_mapping/cube_mapping.h`
- Create: `src/torchscience/csrc/meta/graphics/texture_mapping/cube_mapping.h`
- Create: `src/torchscience/csrc/autograd/graphics/texture_mapping/cube_mapping.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Modify: `src/torchscience/graphics/__init__.py`
- Create: `tests/torchscience/graphics/texture_mapping/__init__.py`
- Create: `tests/torchscience/graphics/texture_mapping/test__cube_mapping.py`

**Cube mapping algorithm:**
```
1. Find face with largest magnitude component
2. Compute UV from remaining components
3. Sample texture at (face, u, v)
```

---

## Task 9: Add `projection` submodule with `perspective_projection`

**Files:**
- Create: `src/torchscience/graphics/projection/__init__.py`
- Create: `src/torchscience/graphics/projection/_perspective_projection.py`
- Create: `src/torchscience/csrc/cpu/graphics/projection/perspective_projection.h`
- Create: `src/torchscience/csrc/meta/graphics/projection/perspective_projection.h`
- Create: `src/torchscience/csrc/autograd/graphics/projection/perspective_projection.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Modify: `src/torchscience/graphics/__init__.py`
- Create: `tests/torchscience/graphics/projection/__init__.py`
- Create: `tests/torchscience/graphics/projection/test__perspective_projection.py`

**Perspective projection matrix (OpenGL):**
```
| f/aspect  0    0              0           |
| 0         f    0              0           |
| 0         0    (f+n)/(n-f)    2fn/(n-f)   |
| 0         0    -1             0           |

where f = 1/tan(fov/2)
```

---

## Task 10: Update graphics __init__.py

**Files:**
- Modify: `src/torchscience/graphics/__init__.py`

```python
from . import color, light_sources, projection, shading, texture_mapping, tone_mapping

__all__ = [
    "color",
    "light_sources",
    "projection",
    "shading",
    "texture_mapping",
    "tone_mapping",
]
```

---

## Task 11: Run full test suite and fix any issues

Run: `uv run pytest tests/torchscience/graphics/ -v`
Expected: All tests PASS

---

## Task 12: Final commit

```bash
git add -A
git commit -m "feat(graphics): complete MVP with 6 submodules and 12 operators"
```
