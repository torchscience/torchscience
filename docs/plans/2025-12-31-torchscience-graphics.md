# torchscience.graphics MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement MVP operators for `torchscience.graphics` covering 6 algorithm areas with 9 new operators.

**Architecture:** Each operator follows the established pattern: Python API -> C++ schema registration -> CPU kernel + Meta implementation + Autograd wrapper. Operators take raw tensors (loose coupling with geometry module).

**Tech Stack:** PyTorch C++ extensions, TORCH_LIBRARY for schema, AT_DISPATCH_FLOATING_TYPES for kernels.

---

## Summary

| Submodule | Operator | Status |
|-----------|----------|--------|
| `color` | `srgb_to_hsv`, `hsv_to_srgb` | Done |
| `shading` | `cook_torrance` | Done |
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
- Create: `src/torchscience/csrc/kernel/graphics/shading/lambertian.h`
- Create: `src/torchscience/csrc/kernel/graphics/shading/lambertian_backward.h`
- Create: `src/torchscience/csrc/cpu/graphics/shading/lambertian.h`
- Create: `src/torchscience/csrc/meta/graphics/shading/lambertian.h`
- Create: `src/torchscience/csrc/autograd/graphics/shading/lambertian.h`
- Modify: `src/torchscience/csrc/torchscience.cpp` (add includes and schema)
- Create: `src/torchscience/graphics/shading/_lambertian.py`
- Modify: `src/torchscience/graphics/shading/__init__.py`
- Create: `tests/torchscience/graphics/shading/test__lambertian.py`

### Step 1: Write the failing test

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

        # At normal incidence, n.l = 1, so result = 1/pi
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

        # n.l = cos(45) ~ 0.707
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

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/torchscience/graphics/shading/test__lambertian.py -v`
Expected: FAIL with "cannot import name 'lambertian'"

### Step 3: Create kernel header (forward)

```cpp
// src/torchscience/csrc/kernel/graphics/shading/lambertian.h
#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::shading {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T dot3(const T* a, const T* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T lambertian(const T* normal, const T* light) {
    const T PI = T(3.14159265358979323846);
    T n_dot_l = dot3(normal, light);
    return n_dot_l > T(0) ? n_dot_l / PI : T(0);
}

}  // namespace torchscience::kernel::graphics::shading
```

### Step 4: Create kernel header (backward)

```cpp
// src/torchscience/csrc/kernel/graphics/shading/lambertian_backward.h
#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::shading {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void lambertian_backward(
    T grad_output,
    const T* normal,
    const T* light,
    T* grad_normal,
    T* grad_light
) {
    const T PI = T(3.14159265358979323846);
    T n_dot_l = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];

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

}  // namespace torchscience::kernel::graphics::shading
```

### Step 5: Create CPU implementation

```cpp
// src/torchscience/csrc/cpu/graphics/shading/lambertian.h
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "kernel/graphics/shading/lambertian.h"
#include "kernel/graphics/shading/lambertian_backward.h"

namespace torchscience::cpu::graphics::shading {

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
                    output_ptr[i] = kernel::graphics::shading::lambertian(
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
                    kernel::graphics::shading::lambertian_backward(
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

### Step 6: Create Meta implementation

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

    // Output shape is batch dimensions (exclude last dim of 3 for vectors)
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

### Step 7: Create Autograd wrapper

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

### Step 8: Add includes and schema to torchscience.cpp

Add these includes (after cook_torrance includes):
```cpp
#include "cpu/graphics/shading/lambertian.h"
#include "autograd/graphics/shading/lambertian.h"
#include "meta/graphics/shading/lambertian.h"
```

Add these schema definitions in TORCH_LIBRARY (after cook_torrance):
```cpp
module.def("lambertian(Tensor normal, Tensor light) -> Tensor");
module.def("lambertian_backward(Tensor grad_output, Tensor normal, Tensor light) -> (Tensor, Tensor)");
```

### Step 9: Create Python API

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
    - Returns 0 when n.l <= 0 (back-facing geometry).
    - Division by pi ensures energy conservation (integrates to 1 over hemisphere).
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

### Step 10: Update shading __init__.py

```python
# src/torchscience/graphics/shading/__init__.py
from ._cook_torrance import cook_torrance
from ._lambertian import lambertian

__all__ = [
    "cook_torrance",
    "lambertian",
]
```

### Step 11: Run tests to verify they pass

Run: `uv run pytest tests/torchscience/graphics/shading/test__lambertian.py -v`
Expected: All tests PASS

### Step 12: Commit

```bash
git add src/torchscience/csrc/kernel/graphics/shading/lambertian.h \
        src/torchscience/csrc/kernel/graphics/shading/lambertian_backward.h \
        src/torchscience/csrc/cpu/graphics/shading/lambertian.h \
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
- Create: `src/torchscience/csrc/kernel/graphics/shading/phong.h`
- Create: `src/torchscience/csrc/kernel/graphics/shading/phong_backward.h`
- Create: `src/torchscience/csrc/cpu/graphics/shading/phong.h`
- Create: `src/torchscience/csrc/meta/graphics/shading/phong.h`
- Create: `src/torchscience/csrc/autograd/graphics/shading/phong.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/graphics/shading/_phong.py`
- Modify: `src/torchscience/graphics/shading/__init__.py`
- Create: `tests/torchscience/graphics/shading/test__phong.py`

### Step 1: Write the failing test

```python
# tests/torchscience/graphics/shading/test__phong.py
"""Tests for Phong specular reflectance."""

import math
import pytest
import torch
from torch.autograd import gradcheck


class TestPhongBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_single_sample(self):
        """Output shape matches batch dimensions for single sample."""
        from torchscience.graphics.shading import phong

        normal = torch.tensor([[0.0, 1.0, 0.0]])
        view = torch.tensor([[0.0, 0.707, 0.707]])
        light = torch.tensor([[0.0, 0.707, -0.707]])
        shininess = torch.tensor([32.0])

        result = phong(normal, view, light, shininess=shininess)

        assert result.shape == (1,)

    def test_output_shape_batch(self):
        """Output shape matches batch dimensions for batched input."""
        from torchscience.graphics.shading import phong

        normal = torch.randn(10, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(10, 3)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(10, 3)
        light = light / light.norm(dim=-1, keepdim=True)
        shininess = torch.full((10,), 32.0)

        result = phong(normal, view, light, shininess=shininess)

        assert result.shape == (10,)

    def test_specular_non_negative(self):
        """Specular values are always non-negative."""
        from torchscience.graphics.shading import phong

        normal = torch.randn(100, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(100, 3)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(100, 3)
        light = light / light.norm(dim=-1, keepdim=True)
        shininess = torch.full((100,), 32.0)

        result = phong(normal, view, light, shininess=shininess)

        assert (result >= 0).all()


class TestPhongCorrectness:
    """Tests for numerical correctness."""

    def test_perfect_reflection(self):
        """Maximum specular when view equals reflection direction."""
        from torchscience.graphics.shading import phong

        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        light = torch.tensor([[0.0, 0.707106781, -0.707106781]], dtype=torch.float64)
        # Reflection of light about normal: R = 2(n.l)n - l
        # n.l = 0.707, so R = 2*0.707*(0,1,0) - (0,0.707,-0.707) = (0, 0.707, 0.707)
        view = torch.tensor([[0.0, 0.707106781, 0.707106781]], dtype=torch.float64)
        shininess = torch.tensor([32.0], dtype=torch.float64)

        result = phong(normal, view, light, shininess=shininess)

        # R.v = 1, so result = 1^32 = 1.0
        torch.testing.assert_close(
            result,
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_back_facing_returns_zero(self):
        """Specular returns 0 when surface is back-facing."""
        from torchscience.graphics.shading import phong

        normal = torch.tensor([[0.0, 1.0, 0.0]])
        view = torch.tensor([[0.0, 0.707, 0.707]])
        light = torch.tensor([[0.0, -0.5, 0.866]])  # Below horizon
        shininess = torch.tensor([32.0])

        result = phong(normal, view, light, shininess=shininess)

        assert result.item() == 0.0


class TestPhongGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Passes gradcheck for basic inputs."""
        from torchscience.graphics.shading import phong

        normal = torch.randn(3, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        normal = normal.detach().requires_grad_(True)

        view = torch.randn(3, 3, dtype=torch.float64)
        view = view / view.norm(dim=-1, keepdim=True)
        view = view.detach().requires_grad_(True)

        light = torch.randn(3, 3, dtype=torch.float64)
        light = light / light.norm(dim=-1, keepdim=True)
        light = light.detach().requires_grad_(True)

        shininess = torch.full((3,), 32.0, dtype=torch.float64, requires_grad=True)

        def func(n, v, l, s):
            return phong(n, v, l, shininess=s)

        assert gradcheck(func, (normal, view, light, shininess), raise_exception=True)


class TestPhongValidation:
    """Tests for input validation."""

    def test_normal_wrong_dimension(self):
        """Raises error when normal last dimension != 3."""
        from torchscience.graphics.shading import phong

        normal = torch.randn(10, 2)
        view = torch.randn(10, 3)
        light = torch.randn(10, 3)
        shininess = torch.full((10,), 32.0)

        with pytest.raises(ValueError, match="normal must have last dimension 3"):
            phong(normal, view, light, shininess=shininess)


class TestPhongDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.shading import phong

        normal = torch.randn(5, 3, dtype=torch.float32)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(5, 3, dtype=torch.float32)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(5, 3, dtype=torch.float32)
        light = light / light.norm(dim=-1, keepdim=True)
        shininess = torch.full((5,), 32.0, dtype=torch.float32)

        result = phong(normal, view, light, shininess=shininess)

        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.graphics.shading import phong

        normal = torch.randn(5, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(5, 3, dtype=torch.float64)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(5, 3, dtype=torch.float64)
        light = light / light.norm(dim=-1, keepdim=True)
        shininess = torch.full((5,), 32.0, dtype=torch.float64)

        result = phong(normal, view, light, shininess=shininess)

        assert result.dtype == torch.float64
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/torchscience/graphics/shading/test__phong.py -v`
Expected: FAIL with "cannot import name 'phong'"

### Step 3: Create kernel header (forward)

```cpp
// src/torchscience/csrc/kernel/graphics/shading/phong.h
#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::shading {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T phong(
    const T* normal,
    const T* view,
    const T* light,
    T shininess
) {
    // n.l
    T n_dot_l = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];

    // Back-facing check
    if (n_dot_l <= T(0)) {
        return T(0);
    }

    // Compute reflection vector: R = 2(n.l)n - l
    T two_n_dot_l = T(2) * n_dot_l;
    T reflect[3] = {
        two_n_dot_l * normal[0] - light[0],
        two_n_dot_l * normal[1] - light[1],
        two_n_dot_l * normal[2] - light[2]
    };

    // R.v
    T r_dot_v = reflect[0] * view[0] + reflect[1] * view[1] + reflect[2] * view[2];

    if (r_dot_v <= T(0)) {
        return T(0);
    }

    // Specular = (R.v)^shininess
    return std::pow(r_dot_v, shininess);
}

}  // namespace torchscience::kernel::graphics::shading
```

### Step 4: Create kernel header (backward)

```cpp
// src/torchscience/csrc/kernel/graphics/shading/phong_backward.h
#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::shading {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void phong_backward(
    T grad_output,
    const T* normal,
    const T* view,
    const T* light,
    T shininess,
    T* grad_normal,
    T* grad_view,
    T* grad_light,
    T* grad_shininess
) {
    // Initialize gradients to zero
    for (int i = 0; i < 3; ++i) {
        grad_normal[i] = T(0);
        grad_view[i] = T(0);
        grad_light[i] = T(0);
    }
    *grad_shininess = T(0);

    T n_dot_l = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];

    if (n_dot_l <= T(0)) {
        return;
    }

    // Reflection vector
    T two_n_dot_l = T(2) * n_dot_l;
    T reflect[3] = {
        two_n_dot_l * normal[0] - light[0],
        two_n_dot_l * normal[1] - light[1],
        two_n_dot_l * normal[2] - light[2]
    };

    T r_dot_v = reflect[0] * view[0] + reflect[1] * view[1] + reflect[2] * view[2];

    if (r_dot_v <= T(0)) {
        return;
    }

    // Forward value: f = (R.v)^shininess
    T f = std::pow(r_dot_v, shininess);

    // df/d(shininess) = f * log(R.v)
    *grad_shininess = grad_output * f * std::log(r_dot_v);

    // df/d(R.v) = shininess * (R.v)^(shininess-1)
    T df_drdotv = grad_output * shininess * std::pow(r_dot_v, shininess - T(1));

    // d(R.v)/dv = R
    for (int i = 0; i < 3; ++i) {
        grad_view[i] = df_drdotv * reflect[i];
    }

    // d(R.v)/dR = v, dR/d(n.l) = 2n, dR/dn = 2(n.l)*I, dR/dl = 2n - I
    // d(R.v)/d(n.l) = 2 * (n.v)
    T n_dot_v = normal[0] * view[0] + normal[1] * view[1] + normal[2] * view[2];
    T df_dndotl = df_drdotv * T(2) * n_dot_v;

    // d(n.l)/dn = l, d(n.l)/dl = n
    for (int i = 0; i < 3; ++i) {
        grad_normal[i] = df_dndotl * light[i] + df_drdotv * T(2) * n_dot_l * view[i];
        grad_light[i] = df_dndotl * normal[i] + df_drdotv * (T(2) * normal[i] * n_dot_v - view[i]);
    }
}

}  // namespace torchscience::kernel::graphics::shading
```

### Step 5: Create CPU implementation

```cpp
// src/torchscience/csrc/cpu/graphics/shading/phong.h
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "kernel/graphics/shading/phong.h"
#include "kernel/graphics/shading/phong_backward.h"

namespace torchscience::cpu::graphics::shading {

inline at::Tensor phong(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    TORCH_CHECK(normal.size(-1) == 3, "phong: normal must have last dimension 3");
    TORCH_CHECK(view.size(-1) == 3, "phong: view must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "phong: light must have last dimension 3");

    auto normal_contig = normal.contiguous();
    auto view_contig = view.contiguous();
    auto light_contig = light.contiguous();
    auto shininess_contig = shininess.contiguous();

    // Compute output shape (batch dimensions)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < normal.dim() - 1; ++i) {
        output_shape.push_back(normal.size(i));
    }

    auto output = at::empty(output_shape, normal.options());
    int64_t num_elements = output.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        normal.scalar_type(), "phong_cpu", [&] {
            const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
            const scalar_t* view_ptr = view_contig.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_contig.data_ptr<scalar_t>();
            const scalar_t* shininess_ptr = shininess_contig.data_ptr<scalar_t>();
            scalar_t* output_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    output_ptr[i] = kernel::graphics::shading::phong(
                        normal_ptr + i * 3,
                        view_ptr + i * 3,
                        light_ptr + i * 3,
                        shininess_ptr[i]
                    );
                }
            });
        }
    );

    return output;
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> phong_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    auto grad_output_contig = grad_output.contiguous();
    auto normal_contig = normal.contiguous();
    auto view_contig = view.contiguous();
    auto light_contig = light.contiguous();
    auto shininess_contig = shininess.contiguous();

    auto grad_normal = at::empty_like(normal);
    auto grad_view = at::empty_like(view);
    auto grad_light = at::empty_like(light);
    auto grad_shininess = at::empty_like(shininess);

    int64_t num_elements = grad_output.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        normal.scalar_type(), "phong_backward_cpu", [&] {
            const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
            const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
            const scalar_t* view_ptr = view_contig.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_contig.data_ptr<scalar_t>();
            const scalar_t* shininess_ptr = shininess_contig.data_ptr<scalar_t>();
            scalar_t* grad_normal_ptr = grad_normal.data_ptr<scalar_t>();
            scalar_t* grad_view_ptr = grad_view.data_ptr<scalar_t>();
            scalar_t* grad_light_ptr = grad_light.data_ptr<scalar_t>();
            scalar_t* grad_shininess_ptr = grad_shininess.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    kernel::graphics::shading::phong_backward(
                        grad_output_ptr[i],
                        normal_ptr + i * 3,
                        view_ptr + i * 3,
                        light_ptr + i * 3,
                        shininess_ptr[i],
                        grad_normal_ptr + i * 3,
                        grad_view_ptr + i * 3,
                        grad_light_ptr + i * 3,
                        grad_shininess_ptr + i
                    );
                }
            });
        }
    );

    return std::make_tuple(grad_normal, grad_view, grad_light, grad_shininess);
}

}  // namespace torchscience::cpu::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("phong", &torchscience::cpu::graphics::shading::phong);
    m.impl("phong_backward", &torchscience::cpu::graphics::shading::phong_backward);
}
```

### Step 6: Create Meta implementation

```cpp
// src/torchscience/csrc/meta/graphics/shading/phong.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::shading {

inline at::Tensor phong(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    TORCH_CHECK(normal.size(-1) == 3, "phong: normal must have last dimension 3");
    TORCH_CHECK(view.size(-1) == 3, "phong: view must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "phong: light must have last dimension 3");

    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < normal.dim() - 1; ++i) {
        output_shape.push_back(normal.size(i));
    }

    return at::empty(output_shape, normal.options());
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> phong_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    return std::make_tuple(
        at::empty_like(normal),
        at::empty_like(view),
        at::empty_like(light),
        at::empty_like(shininess)
    );
}

}  // namespace torchscience::meta::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("phong", &torchscience::meta::graphics::shading::phong);
    m.impl("phong_backward", &torchscience::meta::graphics::shading::phong_backward);
}
```

### Step 7: Create Autograd wrapper

```cpp
// src/torchscience/csrc/autograd/graphics/shading/phong.h
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::graphics::shading {

class Phong : public torch::autograd::Function<Phong> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& normal,
        const at::Tensor& view,
        const at::Tensor& light,
        const at::Tensor& shininess
    ) {
        ctx->saved_data["normal_requires_grad"] = normal.requires_grad();
        ctx->saved_data["view_requires_grad"] = view.requires_grad();
        ctx->saved_data["light_requires_grad"] = light.requires_grad();
        ctx->saved_data["shininess_requires_grad"] = shininess.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::phong", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(normal, view, light, shininess);

        ctx->save_for_backward({normal, view, light, shininess});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor normal = saved[0];
        at::Tensor view = saved[1];
        at::Tensor light = saved[2];
        at::Tensor shininess = saved[3];

        at::Tensor grad_output = grad_outputs[0];

        bool normal_requires_grad = ctx->saved_data["normal_requires_grad"].toBool();
        bool view_requires_grad = ctx->saved_data["view_requires_grad"].toBool();
        bool light_requires_grad = ctx->saved_data["light_requires_grad"].toBool();
        bool shininess_requires_grad = ctx->saved_data["shininess_requires_grad"].toBool();

        if (!normal_requires_grad && !view_requires_grad && !light_requires_grad && !shininess_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_normal, grad_view, grad_light, grad_shininess] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::phong_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output, normal, view, light, shininess);

        return {
            normal_requires_grad ? grad_normal : at::Tensor(),
            view_requires_grad ? grad_view : at::Tensor(),
            light_requires_grad ? grad_light : at::Tensor(),
            shininess_requires_grad ? grad_shininess : at::Tensor()
        };
    }
};

inline at::Tensor phong(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    return Phong::apply(normal, view, light, shininess);
}

}  // namespace torchscience::autograd::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("phong", &torchscience::autograd::graphics::shading::phong);
}
```

### Step 8: Add includes and schema to torchscience.cpp

Add includes (after lambertian includes):
```cpp
#include "cpu/graphics/shading/phong.h"
#include "autograd/graphics/shading/phong.h"
#include "meta/graphics/shading/phong.h"
```

Add schema definitions:
```cpp
module.def("phong(Tensor normal, Tensor view, Tensor light, Tensor shininess) -> Tensor");
module.def("phong_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor shininess) -> (Tensor, Tensor, Tensor, Tensor)");
```

### Step 9: Create Python API

```python
# src/torchscience/graphics/shading/_phong.py
"""Phong specular reflectance implementation."""

from typing import Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def phong(
    normal: Tensor,
    view: Tensor,
    light: Tensor,
    *,
    shininess: Union[Tensor, float],
) -> Tensor:
    r"""Compute Phong specular reflectance.

    Evaluates the Phong specular reflection model using the reflection
    vector and view direction.

    Mathematical Definition
    -----------------------
    The reflection vector is:

    .. math::
        R = 2(n \cdot l)n - l

    The specular term is:

    .. math::
        S = \max(0, R \cdot v)^{shininess}

    Parameters
    ----------
    normal : Tensor, shape (..., 3)
        Surface normal vectors. Must be normalized.
    view : Tensor, shape (..., 3)
        View direction vectors (toward camera). Must be normalized.
    light : Tensor, shape (..., 3)
        Light direction vectors (toward light). Must be normalized.
    shininess : Tensor or float, shape (...) or scalar
        Specular exponent controlling highlight sharpness.
        Higher values produce smaller, sharper highlights.

    Returns
    -------
    Tensor, shape (...)
        Specular reflectance values in [0, 1].

    Examples
    --------
    >>> normal = torch.tensor([[0.0, 1.0, 0.0]])
    >>> view = torch.tensor([[0.0, 0.707, 0.707]])
    >>> light = torch.tensor([[0.0, 0.707, -0.707]])
    >>> torchscience.graphics.shading.phong(normal, view, light, shininess=32.0)
    tensor([...])

    Notes
    -----
    - All direction vectors must be normalized.
    - Returns 0 when n.l <= 0 (back-facing) or R.v <= 0.

    References
    ----------
    .. [1] B.T. Phong, "Illumination for Computer Generated Pictures",
           Communications of the ACM, 1975.
    """
    if normal.shape[-1] != 3:
        raise ValueError(f"normal must have last dimension 3, got {normal.shape[-1]}")
    if view.shape[-1] != 3:
        raise ValueError(f"view must have last dimension 3, got {view.shape[-1]}")
    if light.shape[-1] != 3:
        raise ValueError(f"light must have last dimension 3, got {light.shape[-1]}")

    if not isinstance(shininess, Tensor):
        shininess = torch.tensor(shininess, device=normal.device, dtype=normal.dtype)

    return torch.ops.torchscience.phong(normal, view, light, shininess)
```

### Step 10: Update shading __init__.py

```python
# src/torchscience/graphics/shading/__init__.py
from ._cook_torrance import cook_torrance
from ._lambertian import lambertian
from ._phong import phong

__all__ = [
    "cook_torrance",
    "lambertian",
    "phong",
]
```

### Step 11: Run tests

Run: `uv run pytest tests/torchscience/graphics/shading/test__phong.py -v`
Expected: All tests PASS

### Step 12: Commit

```bash
git add src/torchscience/csrc/kernel/graphics/shading/phong.h \
        src/torchscience/csrc/kernel/graphics/shading/phong_backward.h \
        src/torchscience/csrc/cpu/graphics/shading/phong.h \
        src/torchscience/csrc/meta/graphics/shading/phong.h \
        src/torchscience/csrc/autograd/graphics/shading/phong.h \
        src/torchscience/csrc/torchscience.cpp \
        src/torchscience/graphics/shading/_phong.py \
        src/torchscience/graphics/shading/__init__.py \
        tests/torchscience/graphics/shading/test__phong.py
git commit -m "feat(shading): add phong specular operator"
```

---

## Task 3: Add `blinn_phong` shading operator

**Files:**
- Create: `src/torchscience/csrc/kernel/graphics/shading/blinn_phong.h`
- Create: `src/torchscience/csrc/kernel/graphics/shading/blinn_phong_backward.h`
- Create: `src/torchscience/csrc/cpu/graphics/shading/blinn_phong.h`
- Create: `src/torchscience/csrc/meta/graphics/shading/blinn_phong.h`
- Create: `src/torchscience/csrc/autograd/graphics/shading/blinn_phong.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/graphics/shading/_blinn_phong.py`
- Modify: `src/torchscience/graphics/shading/__init__.py`
- Create: `tests/torchscience/graphics/shading/test__blinn_phong.py`

### Step 1: Write the failing test

```python
# tests/torchscience/graphics/shading/test__blinn_phong.py
"""Tests for Blinn-Phong specular reflectance."""

import math
import pytest
import torch
from torch.autograd import gradcheck


class TestBlinnPhongBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_single_sample(self):
        """Output shape matches batch dimensions for single sample."""
        from torchscience.graphics.shading import blinn_phong

        normal = torch.tensor([[0.0, 1.0, 0.0]])
        view = torch.tensor([[0.0, 0.707, 0.707]])
        light = torch.tensor([[0.0, 0.707, -0.707]])
        shininess = torch.tensor([32.0])

        result = blinn_phong(normal, view, light, shininess=shininess)

        assert result.shape == (1,)

    def test_specular_non_negative(self):
        """Specular values are always non-negative."""
        from torchscience.graphics.shading import blinn_phong

        normal = torch.randn(100, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(100, 3)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(100, 3)
        light = light / light.norm(dim=-1, keepdim=True)
        shininess = torch.full((100,), 32.0)

        result = blinn_phong(normal, view, light, shininess=shininess)

        assert (result >= 0).all()


class TestBlinnPhongCorrectness:
    """Tests for numerical correctness."""

    def test_normal_aligned_with_halfway(self):
        """Maximum specular when normal equals halfway vector."""
        from torchscience.graphics.shading import blinn_phong

        # H = normalize(L + V). If L and V are symmetric about Y axis:
        # L = (0, 0.707, -0.707), V = (0, 0.707, 0.707)
        # H = normalize((0, 1.414, 0)) = (0, 1, 0)
        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        view = torch.tensor([[0.0, 0.707106781, 0.707106781]], dtype=torch.float64)
        light = torch.tensor([[0.0, 0.707106781, -0.707106781]], dtype=torch.float64)
        shininess = torch.tensor([32.0], dtype=torch.float64)

        result = blinn_phong(normal, view, light, shininess=shininess)

        # n.H = 1, so result = 1^32 = 1.0
        torch.testing.assert_close(
            result,
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_back_facing_returns_zero(self):
        """Specular returns 0 when surface is back-facing."""
        from torchscience.graphics.shading import blinn_phong

        normal = torch.tensor([[0.0, 1.0, 0.0]])
        view = torch.tensor([[0.0, 0.707, 0.707]])
        light = torch.tensor([[0.0, -0.5, 0.866]])  # Below horizon
        shininess = torch.tensor([32.0])

        result = blinn_phong(normal, view, light, shininess=shininess)

        assert result.item() == 0.0


class TestBlinnPhongGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Passes gradcheck for basic inputs."""
        from torchscience.graphics.shading import blinn_phong

        normal = torch.randn(3, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        normal = normal.detach().requires_grad_(True)

        view = torch.randn(3, 3, dtype=torch.float64)
        view = view / view.norm(dim=-1, keepdim=True)
        view = view.detach().requires_grad_(True)

        light = torch.randn(3, 3, dtype=torch.float64)
        light = light / light.norm(dim=-1, keepdim=True)
        light = light.detach().requires_grad_(True)

        shininess = torch.full((3,), 32.0, dtype=torch.float64, requires_grad=True)

        def func(n, v, l, s):
            return blinn_phong(n, v, l, shininess=s)

        assert gradcheck(func, (normal, view, light, shininess), raise_exception=True)


class TestBlinnPhongDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.shading import blinn_phong

        normal = torch.randn(5, 3, dtype=torch.float32)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(5, 3, dtype=torch.float32)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(5, 3, dtype=torch.float32)
        light = light / light.norm(dim=-1, keepdim=True)
        shininess = torch.full((5,), 32.0, dtype=torch.float32)

        result = blinn_phong(normal, view, light, shininess=shininess)

        assert result.dtype == torch.float32
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/torchscience/graphics/shading/test__blinn_phong.py -v`
Expected: FAIL with "cannot import name 'blinn_phong'"

### Step 3: Create kernel header (forward)

```cpp
// src/torchscience/csrc/kernel/graphics/shading/blinn_phong.h
#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::shading {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T blinn_phong(
    const T* normal,
    const T* view,
    const T* light,
    T shininess
) {
    // n.l check for back-facing
    T n_dot_l = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];

    if (n_dot_l <= T(0)) {
        return T(0);
    }

    // Compute halfway vector: H = normalize(L + V)
    T h[3] = {
        light[0] + view[0],
        light[1] + view[1],
        light[2] + view[2]
    };
    T h_len = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);

    if (h_len < T(1e-7)) {
        return T(0);
    }

    T inv_h_len = T(1) / h_len;
    h[0] *= inv_h_len;
    h[1] *= inv_h_len;
    h[2] *= inv_h_len;

    // n.H
    T n_dot_h = normal[0] * h[0] + normal[1] * h[1] + normal[2] * h[2];

    if (n_dot_h <= T(0)) {
        return T(0);
    }

    // Specular = (n.H)^shininess
    return std::pow(n_dot_h, shininess);
}

}  // namespace torchscience::kernel::graphics::shading
```

### Step 4: Create kernel header (backward)

```cpp
// src/torchscience/csrc/kernel/graphics/shading/blinn_phong_backward.h
#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::shading {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void blinn_phong_backward(
    T grad_output,
    const T* normal,
    const T* view,
    const T* light,
    T shininess,
    T* grad_normal,
    T* grad_view,
    T* grad_light,
    T* grad_shininess
) {
    for (int i = 0; i < 3; ++i) {
        grad_normal[i] = T(0);
        grad_view[i] = T(0);
        grad_light[i] = T(0);
    }
    *grad_shininess = T(0);

    T n_dot_l = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];

    if (n_dot_l <= T(0)) {
        return;
    }

    // Halfway vector
    T h[3] = { light[0] + view[0], light[1] + view[1], light[2] + view[2] };
    T h_len = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);

    if (h_len < T(1e-7)) {
        return;
    }

    T inv_h_len = T(1) / h_len;
    T h_norm[3] = { h[0] * inv_h_len, h[1] * inv_h_len, h[2] * inv_h_len };

    T n_dot_h = normal[0] * h_norm[0] + normal[1] * h_norm[1] + normal[2] * h_norm[2];

    if (n_dot_h <= T(0)) {
        return;
    }

    T f = std::pow(n_dot_h, shininess);

    // df/d(shininess) = f * log(n.H)
    *grad_shininess = grad_output * f * std::log(n_dot_h);

    // df/d(n.H) = shininess * (n.H)^(shininess-1)
    T df_dndoth = grad_output * shininess * std::pow(n_dot_h, shininess - T(1));

    // d(n.H)/dn = H_normalized
    for (int i = 0; i < 3; ++i) {
        grad_normal[i] = df_dndoth * h_norm[i];
    }

    // d(n.H)/dH = n, then dH/dv = dH/dl = I (before normalization)
    // d(H_norm)/dH = (I - H_norm * H_norm^T) / ||H||
    // d(n.H_norm)/dv = d(n.H_norm)/dl = (n - (n.H)*H_norm) / ||H||
    for (int i = 0; i < 3; ++i) {
        T dndoth_dvi = (normal[i] - n_dot_h * h_norm[i]) * inv_h_len;
        grad_view[i] = df_dndoth * dndoth_dvi;
        grad_light[i] = df_dndoth * dndoth_dvi;
    }
}

}  // namespace torchscience::kernel::graphics::shading
```

### Step 5: Create CPU implementation

```cpp
// src/torchscience/csrc/cpu/graphics/shading/blinn_phong.h
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "kernel/graphics/shading/blinn_phong.h"
#include "kernel/graphics/shading/blinn_phong_backward.h"

namespace torchscience::cpu::graphics::shading {

inline at::Tensor blinn_phong(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    TORCH_CHECK(normal.size(-1) == 3, "blinn_phong: normal must have last dimension 3");
    TORCH_CHECK(view.size(-1) == 3, "blinn_phong: view must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "blinn_phong: light must have last dimension 3");

    auto normal_contig = normal.contiguous();
    auto view_contig = view.contiguous();
    auto light_contig = light.contiguous();
    auto shininess_contig = shininess.contiguous();

    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < normal.dim() - 1; ++i) {
        output_shape.push_back(normal.size(i));
    }

    auto output = at::empty(output_shape, normal.options());
    int64_t num_elements = output.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        normal.scalar_type(), "blinn_phong_cpu", [&] {
            const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
            const scalar_t* view_ptr = view_contig.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_contig.data_ptr<scalar_t>();
            const scalar_t* shininess_ptr = shininess_contig.data_ptr<scalar_t>();
            scalar_t* output_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    output_ptr[i] = kernel::graphics::shading::blinn_phong(
                        normal_ptr + i * 3,
                        view_ptr + i * 3,
                        light_ptr + i * 3,
                        shininess_ptr[i]
                    );
                }
            });
        }
    );

    return output;
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> blinn_phong_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    auto grad_output_contig = grad_output.contiguous();
    auto normal_contig = normal.contiguous();
    auto view_contig = view.contiguous();
    auto light_contig = light.contiguous();
    auto shininess_contig = shininess.contiguous();

    auto grad_normal = at::empty_like(normal);
    auto grad_view = at::empty_like(view);
    auto grad_light = at::empty_like(light);
    auto grad_shininess = at::empty_like(shininess);

    int64_t num_elements = grad_output.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        normal.scalar_type(), "blinn_phong_backward_cpu", [&] {
            const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
            const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
            const scalar_t* view_ptr = view_contig.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_contig.data_ptr<scalar_t>();
            const scalar_t* shininess_ptr = shininess_contig.data_ptr<scalar_t>();
            scalar_t* grad_normal_ptr = grad_normal.data_ptr<scalar_t>();
            scalar_t* grad_view_ptr = grad_view.data_ptr<scalar_t>();
            scalar_t* grad_light_ptr = grad_light.data_ptr<scalar_t>();
            scalar_t* grad_shininess_ptr = grad_shininess.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    kernel::graphics::shading::blinn_phong_backward(
                        grad_output_ptr[i],
                        normal_ptr + i * 3,
                        view_ptr + i * 3,
                        light_ptr + i * 3,
                        shininess_ptr[i],
                        grad_normal_ptr + i * 3,
                        grad_view_ptr + i * 3,
                        grad_light_ptr + i * 3,
                        grad_shininess_ptr + i
                    );
                }
            });
        }
    );

    return std::make_tuple(grad_normal, grad_view, grad_light, grad_shininess);
}

}  // namespace torchscience::cpu::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("blinn_phong", &torchscience::cpu::graphics::shading::blinn_phong);
    m.impl("blinn_phong_backward", &torchscience::cpu::graphics::shading::blinn_phong_backward);
}
```

### Step 6: Create Meta implementation

```cpp
// src/torchscience/csrc/meta/graphics/shading/blinn_phong.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::shading {

inline at::Tensor blinn_phong(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    TORCH_CHECK(normal.size(-1) == 3, "blinn_phong: normal must have last dimension 3");
    TORCH_CHECK(view.size(-1) == 3, "blinn_phong: view must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "blinn_phong: light must have last dimension 3");

    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < normal.dim() - 1; ++i) {
        output_shape.push_back(normal.size(i));
    }

    return at::empty(output_shape, normal.options());
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> blinn_phong_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    return std::make_tuple(
        at::empty_like(normal),
        at::empty_like(view),
        at::empty_like(light),
        at::empty_like(shininess)
    );
}

}  // namespace torchscience::meta::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("blinn_phong", &torchscience::meta::graphics::shading::blinn_phong);
    m.impl("blinn_phong_backward", &torchscience::meta::graphics::shading::blinn_phong_backward);
}
```

### Step 7: Create Autograd wrapper

```cpp
// src/torchscience/csrc/autograd/graphics/shading/blinn_phong.h
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::graphics::shading {

class BlinnPhong : public torch::autograd::Function<BlinnPhong> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& normal,
        const at::Tensor& view,
        const at::Tensor& light,
        const at::Tensor& shininess
    ) {
        ctx->saved_data["normal_requires_grad"] = normal.requires_grad();
        ctx->saved_data["view_requires_grad"] = view.requires_grad();
        ctx->saved_data["light_requires_grad"] = light.requires_grad();
        ctx->saved_data["shininess_requires_grad"] = shininess.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::blinn_phong", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(normal, view, light, shininess);

        ctx->save_for_backward({normal, view, light, shininess});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor normal = saved[0];
        at::Tensor view = saved[1];
        at::Tensor light = saved[2];
        at::Tensor shininess = saved[3];

        at::Tensor grad_output = grad_outputs[0];

        bool normal_requires_grad = ctx->saved_data["normal_requires_grad"].toBool();
        bool view_requires_grad = ctx->saved_data["view_requires_grad"].toBool();
        bool light_requires_grad = ctx->saved_data["light_requires_grad"].toBool();
        bool shininess_requires_grad = ctx->saved_data["shininess_requires_grad"].toBool();

        if (!normal_requires_grad && !view_requires_grad && !light_requires_grad && !shininess_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_normal, grad_view, grad_light, grad_shininess] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::blinn_phong_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output, normal, view, light, shininess);

        return {
            normal_requires_grad ? grad_normal : at::Tensor(),
            view_requires_grad ? grad_view : at::Tensor(),
            light_requires_grad ? grad_light : at::Tensor(),
            shininess_requires_grad ? grad_shininess : at::Tensor()
        };
    }
};

inline at::Tensor blinn_phong(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    return BlinnPhong::apply(normal, view, light, shininess);
}

}  // namespace torchscience::autograd::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("blinn_phong", &torchscience::autograd::graphics::shading::blinn_phong);
}
```

### Step 8: Add includes and schema to torchscience.cpp

Add includes:
```cpp
#include "cpu/graphics/shading/blinn_phong.h"
#include "autograd/graphics/shading/blinn_phong.h"
#include "meta/graphics/shading/blinn_phong.h"
```

Add schema:
```cpp
module.def("blinn_phong(Tensor normal, Tensor view, Tensor light, Tensor shininess) -> Tensor");
module.def("blinn_phong_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor shininess) -> (Tensor, Tensor, Tensor, Tensor)");
```

### Step 9: Create Python API

```python
# src/torchscience/graphics/shading/_blinn_phong.py
"""Blinn-Phong specular reflectance implementation."""

from typing import Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def blinn_phong(
    normal: Tensor,
    view: Tensor,
    light: Tensor,
    *,
    shininess: Union[Tensor, float],
) -> Tensor:
    r"""Compute Blinn-Phong specular reflectance.

    Evaluates the Blinn-Phong specular model using the halfway vector
    between view and light directions.

    Mathematical Definition
    -----------------------
    The halfway vector is:

    .. math::
        H = \frac{L + V}{||L + V||}

    The specular term is:

    .. math::
        S = \max(0, n \cdot H)^{shininess}

    Parameters
    ----------
    normal : Tensor, shape (..., 3)
        Surface normal vectors. Must be normalized.
    view : Tensor, shape (..., 3)
        View direction vectors (toward camera). Must be normalized.
    light : Tensor, shape (..., 3)
        Light direction vectors (toward light). Must be normalized.
    shininess : Tensor or float, shape (...) or scalar
        Specular exponent controlling highlight sharpness.

    Returns
    -------
    Tensor, shape (...)
        Specular reflectance values in [0, 1].

    Notes
    -----
    Blinn-Phong is more efficient than Phong as it avoids computing
    the reflection vector. It produces similar results with
    shininess values approximately 4x higher than Phong.

    References
    ----------
    .. [1] J.F. Blinn, "Models of Light Reflection for Computer
           Synthesized Pictures", SIGGRAPH 1977.
    """
    if normal.shape[-1] != 3:
        raise ValueError(f"normal must have last dimension 3, got {normal.shape[-1]}")
    if view.shape[-1] != 3:
        raise ValueError(f"view must have last dimension 3, got {view.shape[-1]}")
    if light.shape[-1] != 3:
        raise ValueError(f"light must have last dimension 3, got {light.shape[-1]}")

    if not isinstance(shininess, Tensor):
        shininess = torch.tensor(shininess, device=normal.device, dtype=normal.dtype)

    return torch.ops.torchscience.blinn_phong(normal, view, light, shininess)
```

### Step 10: Update shading __init__.py

```python
# src/torchscience/graphics/shading/__init__.py
from ._blinn_phong import blinn_phong
from ._cook_torrance import cook_torrance
from ._lambertian import lambertian
from ._phong import phong

__all__ = [
    "blinn_phong",
    "cook_torrance",
    "lambertian",
    "phong",
]
```

### Step 11: Run tests

Run: `uv run pytest tests/torchscience/graphics/shading/test__blinn_phong.py -v`
Expected: All tests PASS

### Step 12: Commit

```bash
git add src/torchscience/csrc/kernel/graphics/shading/blinn_phong.h \
        src/torchscience/csrc/kernel/graphics/shading/blinn_phong_backward.h \
        src/torchscience/csrc/cpu/graphics/shading/blinn_phong.h \
        src/torchscience/csrc/meta/graphics/shading/blinn_phong.h \
        src/torchscience/csrc/autograd/graphics/shading/blinn_phong.h \
        src/torchscience/csrc/torchscience.cpp \
        src/torchscience/graphics/shading/_blinn_phong.py \
        src/torchscience/graphics/shading/__init__.py \
        tests/torchscience/graphics/shading/test__blinn_phong.py
git commit -m "feat(shading): add blinn_phong specular operator"
```

---

## Task 4: Add `light_sources` submodule with `point_light`

**Files:**
- Create: `src/torchscience/graphics/light_sources/__init__.py`
- Create: `src/torchscience/graphics/light_sources/_point_light.py`
- Create: `src/torchscience/csrc/kernel/graphics/light_sources/point_light.h`
- Create: `src/torchscience/csrc/kernel/graphics/light_sources/point_light_backward.h`
- Create: `src/torchscience/csrc/cpu/graphics/light_sources/point_light.h`
- Create: `src/torchscience/csrc/meta/graphics/light_sources/point_light.h`
- Create: `src/torchscience/csrc/autograd/graphics/light_sources/point_light.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Modify: `src/torchscience/graphics/__init__.py`
- Create: `tests/torchscience/graphics/light_sources/__init__.py`
- Create: `tests/torchscience/graphics/light_sources/test__point_light.py`

### Step 1: Write the failing test

```python
# tests/torchscience/graphics/light_sources/test__point_light.py
"""Tests for point light source."""

import math
import pytest
import torch
from torch.autograd import gradcheck


class TestPointLightBasic:
    """Tests for basic shape and property verification."""

    def test_output_shapes(self):
        """Output shapes are correct."""
        from torchscience.graphics.light_sources import point_light

        light_pos = torch.tensor([[5.0, 5.0, 5.0]])
        surface_pos = torch.tensor([[0.0, 0.0, 0.0]])
        intensity = torch.tensor([100.0])

        irradiance, direction = point_light(light_pos, surface_pos, intensity)

        assert irradiance.shape == (1,)
        assert direction.shape == (1, 3)

    def test_direction_is_normalized(self):
        """Light direction is unit length."""
        from torchscience.graphics.light_sources import point_light

        light_pos = torch.tensor([[5.0, 5.0, 5.0]])
        surface_pos = torch.tensor([[0.0, 0.0, 0.0]])
        intensity = torch.tensor([100.0])

        _, direction = point_light(light_pos, surface_pos, intensity)

        norm = direction.norm(dim=-1)
        torch.testing.assert_close(norm, torch.ones(1), rtol=1e-5, atol=1e-7)


class TestPointLightCorrectness:
    """Tests for numerical correctness."""

    def test_inverse_square_falloff(self):
        """Irradiance follows inverse square law."""
        from torchscience.graphics.light_sources import point_light

        light_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        intensity = torch.tensor([100.0], dtype=torch.float64)

        # At distance 1
        surface_pos_1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        irr_1, _ = point_light(light_pos, surface_pos_1, intensity)

        # At distance 2
        surface_pos_2 = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
        irr_2, _ = point_light(light_pos, surface_pos_2, intensity)

        # irr_2 should be 1/4 of irr_1
        torch.testing.assert_close(irr_2, irr_1 / 4.0, rtol=1e-5, atol=1e-7)

    def test_irradiance_value(self):
        """Irradiance = intensity / distance^2."""
        from torchscience.graphics.light_sources import point_light

        light_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        surface_pos = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float64)  # distance = 5
        intensity = torch.tensor([100.0], dtype=torch.float64)

        irradiance, _ = point_light(light_pos, surface_pos, intensity)

        expected = 100.0 / 25.0  # 100 / 5^2 = 4
        torch.testing.assert_close(
            irradiance,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-7,
        )


class TestPointLightGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Passes gradcheck."""
        from torchscience.graphics.light_sources import point_light

        light_pos = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        surface_pos = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        intensity = torch.randn(3, dtype=torch.float64).abs() + 1.0
        intensity = intensity.detach().requires_grad_(True)

        def func(lp, sp, i):
            irr, _ = point_light(lp, sp, i)
            return irr

        assert gradcheck(func, (light_pos, surface_pos, intensity), raise_exception=True)
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/torchscience/graphics/light_sources/test__point_light.py -v`
Expected: FAIL

### Step 3: Create kernel header (forward)

```cpp
// src/torchscience/csrc/kernel/graphics/light_sources/point_light.h
#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::light_sources {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void point_light(
    const T* light_pos,
    const T* surface_pos,
    T intensity,
    T* irradiance,
    T* direction
) {
    // Direction from surface to light
    T dx = light_pos[0] - surface_pos[0];
    T dy = light_pos[1] - surface_pos[1];
    T dz = light_pos[2] - surface_pos[2];

    T dist_sq = dx * dx + dy * dy + dz * dz;
    T dist = std::sqrt(dist_sq);

    // Avoid division by zero
    if (dist < T(1e-7)) {
        *irradiance = T(0);
        direction[0] = T(0);
        direction[1] = T(1);
        direction[2] = T(0);
        return;
    }

    T inv_dist = T(1) / dist;
    direction[0] = dx * inv_dist;
    direction[1] = dy * inv_dist;
    direction[2] = dz * inv_dist;

    // Inverse square falloff
    *irradiance = intensity / dist_sq;
}

}  // namespace torchscience::kernel::graphics::light_sources
```

### Step 4: Create kernel header (backward)

```cpp
// src/torchscience/csrc/kernel/graphics/light_sources/point_light_backward.h
#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::light_sources {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void point_light_backward(
    T grad_irradiance,
    const T* grad_direction,
    const T* light_pos,
    const T* surface_pos,
    T intensity,
    T* grad_light_pos,
    T* grad_surface_pos,
    T* grad_intensity
) {
    T dx = light_pos[0] - surface_pos[0];
    T dy = light_pos[1] - surface_pos[1];
    T dz = light_pos[2] - surface_pos[2];

    T dist_sq = dx * dx + dy * dy + dz * dz;
    T dist = std::sqrt(dist_sq);

    if (dist < T(1e-7)) {
        for (int i = 0; i < 3; ++i) {
            grad_light_pos[i] = T(0);
            grad_surface_pos[i] = T(0);
        }
        *grad_intensity = T(0);
        return;
    }

    T inv_dist = T(1) / dist;
    T inv_dist_sq = inv_dist * inv_dist;
    T inv_dist_cubed = inv_dist_sq * inv_dist;

    // d(irradiance)/d(intensity) = 1/dist^2
    *grad_intensity = grad_irradiance * inv_dist_sq;

    // d(irradiance)/d(dist^2) = -intensity/dist^4
    T d_irr_d_dist_sq = -intensity * inv_dist_sq * inv_dist_sq;

    // d(dist^2)/d(delta) = 2*delta
    T d_dist_sq[3] = { T(2) * dx, T(2) * dy, T(2) * dz };

    // d(direction)/d(delta) = (I - dir*dir^T) / dist
    T dir[3] = { dx * inv_dist, dy * inv_dist, dz * inv_dist };

    for (int i = 0; i < 3; ++i) {
        // Irradiance gradient contribution
        T grad_from_irr = grad_irradiance * d_irr_d_dist_sq * d_dist_sq[i];

        // Direction gradient contribution: d(dir_i)/d(delta_j)
        T grad_from_dir = T(0);
        for (int j = 0; j < 3; ++j) {
            T kronecker = (i == j) ? T(1) : T(0);
            T d_dir_j_d_delta_i = (kronecker - dir[j] * dir[i]) * inv_dist;
            grad_from_dir += grad_direction[j] * d_dir_j_d_delta_i;
        }

        grad_light_pos[i] = grad_from_irr + grad_from_dir;
        grad_surface_pos[i] = -grad_light_pos[i];
    }
}

}  // namespace torchscience::kernel::graphics::light_sources
```

### Step 5-12: Follow same pattern as shading operators

Create CPU, Meta, Autograd implementations following the same structure.

**Schema:**
```cpp
module.def("point_light(Tensor light_pos, Tensor surface_pos, Tensor intensity) -> (Tensor, Tensor)");
module.def("point_light_backward(Tensor grad_irradiance, Tensor grad_direction, Tensor light_pos, Tensor surface_pos, Tensor intensity) -> (Tensor, Tensor, Tensor)");
```

**Python API:**
```python
# src/torchscience/graphics/light_sources/_point_light.py
"""Point light source implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def point_light(
    light_position: Tensor,
    surface_position: Tensor,
    intensity: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Compute point light irradiance and direction.

    Parameters
    ----------
    light_position : Tensor, shape (..., 3)
        Position of the light source.
    surface_position : Tensor, shape (..., 3)
        Position of the surface point.
    intensity : Tensor, shape (...)
        Light intensity (radiant intensity in watts/sr).

    Returns
    -------
    irradiance : Tensor, shape (...)
        Irradiance at the surface (watts/m^2).
    direction : Tensor, shape (..., 3)
        Normalized direction from surface to light.

    Notes
    -----
    Irradiance follows inverse square law: I / d^2
    """
    return torch.ops.torchscience.point_light(light_position, surface_position, intensity)
```

**Update graphics __init__.py:**
```python
from . import color, light_sources, shading

__all__ = ["color", "light_sources", "shading"]
```

### Commit

```bash
git commit -m "feat(light_sources): add point_light operator"
```

---

## Task 5-11: Remaining operators

Tasks 5-11 follow the same pattern. For brevity, here are the key formulas and schemas:

### Task 5: `spotlight`
**Formula:**
```
theta = acos(dot(-light_direction, spot_direction))
falloff = smoothstep(cos(outer_angle), cos(inner_angle), cos(theta))
irradiance = point_light_irradiance * falloff
```

**Schema:**
```cpp
module.def("spotlight(Tensor light_pos, Tensor surface_pos, Tensor spot_direction, Tensor intensity, Tensor inner_angle, Tensor outer_angle) -> (Tensor, Tensor)");
```

### Task 6: `directional_light`
**Formula:**
```
irradiance = intensity  (constant)
direction = -light_direction  (negated for "toward light" convention)
```

**Schema:**
```cpp
module.def("directional_light(Tensor direction, Tensor intensity) -> (Tensor, Tensor)");
```

### Task 7: `reinhard` tone mapping
**Formula:**
```
Basic: L_out = L_in / (1 + L_in)
Extended: L_out = L_in * (1 + L_in/L_white^2) / (1 + L_in)
```

**Schema:**
```cpp
module.def("reinhard(Tensor input, Tensor? white_point) -> Tensor");
```

### Task 8: `cube_mapping`
**Algorithm:**
```
1. Find face with largest magnitude component (|x|, |y|, or |z|)
2. Compute UV from remaining components
3. Return (face_index, u, v)
```

**Schema:**
```cpp
module.def("cube_mapping(Tensor direction) -> (Tensor, Tensor, Tensor)");
```

### Task 9: `perspective_projection`
**Matrix:**
```
| f/aspect  0    0              0           |
| 0         f    0              0           |
| 0         0    (f+n)/(n-f)    2fn/(n-f)   |
| 0         0    -1             0           |

where f = 1/tan(fov/2)
```

**Schema:**
```cpp
module.def("perspective_projection(Tensor fov, Tensor aspect, Tensor near, Tensor far) -> Tensor");
```

---

## Task 10: Update graphics __init__.py

```python
# src/torchscience/graphics/__init__.py
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

## Task 11: Run full test suite

Run: `uv run pytest tests/torchscience/graphics/ -v`
Expected: All tests PASS

---

## Task 12: Final commit

```bash
git add -A
git commit -m "feat(graphics): complete MVP with 6 submodules and 12 operators"
```
