# torchscience.graphics MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement MVP operators for `torchscience.graphics` covering 5 new operators across shading, lighting, tone mapping, texture, and projection.

**Architecture:** Each operator follows the established pattern: Python API -> C++ schema registration -> CPU kernel + Meta implementation + Autograd wrapper. Operators take raw tensors (loose coupling with geometry module).

**Tech Stack:** PyTorch C++ extensions, TORCH_LIBRARY for schema, AT_DISPATCH_FLOATING_TYPES for kernels.

---

## Summary

| Submodule | Operator | Status |
|-----------|----------|--------|
| `color` | `srgb_to_hsv`, `hsv_to_srgb` | Done |
| `shading` | `cook_torrance` | Done |
| `shading` | `phong` | Task 1 |
| `lighting` | `spotlight` | Task 2 |
| `tone_mapping` | `reinhard` | Task 3 |
| `texture_mapping` | `cube_mapping` | Task 4 |
| `projection` | `perspective_projection` | Task 5 |

---

## Task 1: Add `phong` specular shading operator

**Goal:** Implement Phong specular reflectance model: `S = max(0, R·v)^shininess` where `R = 2(n·l)n - l`

**Files:**
- Create: `tests/torchscience/graphics/shading/test__phong.py`
- Create: `src/torchscience/csrc/kernel/graphics/shading/phong.h`
- Create: `src/torchscience/csrc/kernel/graphics/shading/phong_backward.h`
- Create: `src/torchscience/csrc/cpu/graphics/shading/phong.h`
- Create: `src/torchscience/csrc/meta/graphics/shading/phong.h`
- Create: `src/torchscience/csrc/autograd/graphics/shading/phong.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/graphics/shading/_phong.py`
- Modify: `src/torchscience/graphics/shading/__init__.py`

### Step 1.1: Write the failing test

Create `tests/torchscience/graphics/shading/test__phong.py`:

```python
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

        torch.manual_seed(42)
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

    def test_back_facing_light_returns_zero(self):
        """Specular returns 0 when light is below surface."""
        from torchscience.graphics.shading import phong

        normal = torch.tensor([[0.0, 1.0, 0.0]])
        view = torch.tensor([[0.0, 0.707, 0.707]])
        light = torch.tensor([[0.0, -0.5, 0.866]])  # Below horizon (n.l < 0)
        shininess = torch.tensor([32.0])

        result = phong(normal, view, light, shininess=shininess)

        assert result.item() == 0.0

    def test_off_specular_lower_value(self):
        """Off-specular directions have lower values than mirror direction."""
        from torchscience.graphics.shading import phong

        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        light = torch.tensor([[0.0, 0.707, -0.707]], dtype=torch.float64)
        # Mirror reflection
        n_dot_l = (normal * light).sum(dim=-1, keepdim=True)
        view_mirror = 2 * n_dot_l * normal - light
        # Off-specular view
        view_off = torch.tensor([[0.3, 0.7, 0.648]], dtype=torch.float64)
        view_off = view_off / view_off.norm(dim=-1, keepdim=True)
        shininess = torch.tensor([32.0], dtype=torch.float64)

        result_mirror = phong(normal, view_mirror, light, shininess=shininess)
        result_off = phong(normal, view_off, light, shininess=shininess)

        assert result_mirror.item() > result_off.item()


class TestPhongGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Passes gradcheck for basic inputs."""
        from torchscience.graphics.shading import phong

        torch.manual_seed(123)
        normal = torch.randn(3, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        normal = normal.detach().requires_grad_(True)

        view = torch.randn(3, 3, dtype=torch.float64)
        view = view / view.norm(dim=-1, keepdim=True)
        view = view.detach().requires_grad_(True)

        light = torch.randn(3, 3, dtype=torch.float64)
        light = light / light.norm(dim=-1, keepdim=True)
        light = light.detach().requires_grad_(True)

        shininess = torch.tensor([32.0, 16.0, 64.0], dtype=torch.float64, requires_grad=True)

        def func(n, v, l, s):
            return phong(n, v, l, shininess=s)

        assert gradcheck(func, (normal, view, light, shininess), raise_exception=True)

    def test_gradients_finite(self):
        """Gradients are finite for typical inputs."""
        from torchscience.graphics.shading import phong

        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
        view = torch.tensor([[0.0, 0.707, 0.707]], dtype=torch.float64, requires_grad=True)
        light = torch.tensor([[0.0, 0.707, -0.707]], dtype=torch.float64, requires_grad=True)
        shininess = torch.tensor([32.0], dtype=torch.float64, requires_grad=True)

        result = phong(normal, view, light, shininess=shininess)
        result.sum().backward()

        assert normal.grad is not None and torch.isfinite(normal.grad).all()
        assert view.grad is not None and torch.isfinite(view.grad).all()
        assert light.grad is not None and torch.isfinite(light.grad).all()
        assert shininess.grad is not None and torch.isfinite(shininess.grad).all()


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

### Step 1.2: Run test to verify it fails

Run: `uv run pytest tests/torchscience/graphics/shading/test__phong.py -v`
Expected: FAIL with "cannot import name 'phong'"

### Step 1.3: Create kernel header (forward)

Create `src/torchscience/csrc/kernel/graphics/shading/phong.h`:

```cpp
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

### Step 1.4: Create kernel header (backward)

Create `src/torchscience/csrc/kernel/graphics/shading/phong_backward.h`:

```cpp
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

    // d(R.v)/dR = v, dR/d(n.l) = 2n, dR/dn = 2(n.l)*I, dR/dl = 2n*n^T - I
    // d(R.v)/d(n.l) = 2 * (n.v)
    T n_dot_v = normal[0] * view[0] + normal[1] * view[1] + normal[2] * view[2];
    T df_dndotl = df_drdotv * T(2) * n_dot_v;

    // d(n.l)/dn = l, d(n.l)/dl = n
    // Total grad_normal = df/d(n.l) * l + df/dR * dR/dn
    // dR/dn = 2(n.l)*I (only the diagonal part matters for dot product)
    // Actually: R_i = 2(n.l)*n_i - l_i
    // dR_i/dn_j = 2*l_j*n_i + 2*(n.l)*delta_ij
    // d(R.v)/dn_j = sum_i v_i * dR_i/dn_j = 2*l_j*(n.v) + 2*(n.l)*v_j
    for (int i = 0; i < 3; ++i) {
        grad_normal[i] = df_drdotv * (T(2) * light[i] * n_dot_v + T(2) * n_dot_l * view[i]);
        // dR_i/dl_j = 2*n_j*n_i - delta_ij
        // d(R.v)/dl_j = sum_i v_i * dR_i/dl_j = 2*n_j*(n.v) - v_j
        grad_light[i] = df_drdotv * (T(2) * normal[i] * n_dot_v - view[i]);
    }
}

}  // namespace torchscience::kernel::graphics::shading
```

### Step 1.5: Create CPU implementation

Create `src/torchscience/csrc/cpu/graphics/shading/phong.h`:

```cpp
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

### Step 1.6: Create Meta implementation

Create `src/torchscience/csrc/meta/graphics/shading/phong.h`:

```cpp
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

### Step 1.7: Create Autograd wrapper

Create `src/torchscience/csrc/autograd/graphics/shading/phong.h`:

```cpp
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

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

### Step 1.8: Add includes and schema to torchscience.cpp

Add include after other graphics includes:
```cpp
#include "cpu/graphics/shading/phong.h"
#include "autograd/graphics/shading/phong.h"
#include "meta/graphics/shading/phong.h"
```

Add schema definitions in TORCH_LIBRARY block:
```cpp
// Phong shading
module.def("phong(Tensor normal, Tensor view, Tensor light, Tensor shininess) -> Tensor");
module.def("phong_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor shininess) -> (Tensor, Tensor, Tensor, Tensor)");
```

### Step 1.9: Create Python API

Create `src/torchscience/graphics/shading/_phong.py`:

```python
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

### Step 1.10: Update shading __init__.py

Modify `src/torchscience/graphics/shading/__init__.py`:

```python
from ._cook_torrance import cook_torrance
from ._phong import phong

__all__ = [
    "cook_torrance",
    "phong",
]
```

### Step 1.11: Run tests

Run: `uv run pytest tests/torchscience/graphics/shading/test__phong.py -v`
Expected: All tests PASS

### Step 1.12: Commit

```bash
git add tests/torchscience/graphics/shading/test__phong.py \
        src/torchscience/csrc/kernel/graphics/shading/phong.h \
        src/torchscience/csrc/kernel/graphics/shading/phong_backward.h \
        src/torchscience/csrc/cpu/graphics/shading/phong.h \
        src/torchscience/csrc/meta/graphics/shading/phong.h \
        src/torchscience/csrc/autograd/graphics/shading/phong.h \
        src/torchscience/csrc/torchscience.cpp \
        src/torchscience/graphics/shading/_phong.py \
        src/torchscience/graphics/shading/__init__.py
git commit -m "feat(shading): add phong specular operator"
```

---

## Task 2: Add `spotlight` operator

**Goal:** Implement spotlight attenuation with angular falloff.

**Formula:**
```
theta = acos(dot(-light_to_surface, spot_direction))
falloff = smoothstep(cos(outer_angle), cos(inner_angle), cos(theta))
irradiance = intensity * falloff / distance^2
```

**Files:**
- Create: `tests/torchscience/graphics/lighting/test__spotlight.py`
- Create: `src/torchscience/csrc/kernel/graphics/lighting/spotlight.h`
- Create: `src/torchscience/csrc/kernel/graphics/lighting/spotlight_backward.h`
- Create: `src/torchscience/csrc/cpu/graphics/lighting/spotlight.h`
- Create: `src/torchscience/csrc/meta/graphics/lighting/spotlight.h`
- Create: `src/torchscience/csrc/autograd/graphics/lighting/spotlight.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/graphics/lighting/__init__.py`
- Create: `src/torchscience/graphics/lighting/_spotlight.py`
- Modify: `src/torchscience/graphics/__init__.py`

### Step 2.1: Write the failing test

Create `tests/torchscience/graphics/lighting/__init__.py` (empty file).

Create `tests/torchscience/graphics/lighting/test__spotlight.py`:

```python
"""Tests for spotlight light source."""

import math
import pytest
import torch
from torch.autograd import gradcheck


class TestSpotlightBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_single_sample(self):
        """Output has correct shape for single sample."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.tensor([[0.0, 5.0, 0.0]])
        surface_pos = torch.tensor([[0.0, 0.0, 0.0]])
        spot_direction = torch.tensor([[0.0, -1.0, 0.0]])
        intensity = torch.tensor([100.0])
        inner_angle = torch.tensor([math.radians(15)])
        outer_angle = torch.tensor([math.radians(30)])

        irradiance, light_dir = spotlight(
            light_pos, surface_pos, spot_direction,
            intensity=intensity, inner_angle=inner_angle, outer_angle=outer_angle
        )

        assert irradiance.shape == (1,)
        assert light_dir.shape == (1, 3)

    def test_output_shape_batch(self):
        """Output has correct shape for batched input."""
        from torchscience.graphics.lighting import spotlight

        batch = 10
        light_pos = torch.randn(batch, 3)
        surface_pos = torch.randn(batch, 3)
        spot_direction = torch.randn(batch, 3)
        spot_direction = spot_direction / spot_direction.norm(dim=-1, keepdim=True)
        intensity = torch.full((batch,), 100.0)
        inner_angle = torch.full((batch,), math.radians(15))
        outer_angle = torch.full((batch,), math.radians(30))

        irradiance, light_dir = spotlight(
            light_pos, surface_pos, spot_direction,
            intensity=intensity, inner_angle=inner_angle, outer_angle=outer_angle
        )

        assert irradiance.shape == (batch,)
        assert light_dir.shape == (batch, 3)

    def test_irradiance_non_negative(self):
        """Irradiance is always non-negative."""
        from torchscience.graphics.lighting import spotlight

        torch.manual_seed(42)
        batch = 100
        light_pos = torch.randn(batch, 3)
        surface_pos = torch.randn(batch, 3)
        spot_direction = torch.randn(batch, 3)
        spot_direction = spot_direction / spot_direction.norm(dim=-1, keepdim=True)
        intensity = torch.full((batch,), 100.0)
        inner_angle = torch.full((batch,), math.radians(15))
        outer_angle = torch.full((batch,), math.radians(30))

        irradiance, _ = spotlight(
            light_pos, surface_pos, spot_direction,
            intensity=intensity, inner_angle=inner_angle, outer_angle=outer_angle
        )

        assert (irradiance >= 0).all()


class TestSpotlightCorrectness:
    """Tests for numerical correctness."""

    def test_inside_inner_cone_full_intensity(self):
        """Point inside inner cone receives full intensity (before distance falloff)."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64)
        surface_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)  # Directly below
        spot_direction = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)  # Pointing down
        intensity = torch.tensor([100.0], dtype=torch.float64)
        inner_angle = torch.tensor([math.radians(30)], dtype=torch.float64)
        outer_angle = torch.tensor([math.radians(45)], dtype=torch.float64)

        irradiance, _ = spotlight(
            light_pos, surface_pos, spot_direction,
            intensity=intensity, inner_angle=inner_angle, outer_angle=outer_angle
        )

        # Distance is 5, so irradiance = 100 / 25 = 4.0 (full intensity, no angular falloff)
        expected = 100.0 / 25.0
        torch.testing.assert_close(
            irradiance,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-5, atol=1e-7
        )

    def test_outside_outer_cone_zero(self):
        """Point outside outer cone receives zero irradiance."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64)
        # Surface at 60 degrees from spotlight direction (beyond outer cone)
        surface_pos = torch.tensor([[8.66, 0.0, 0.0]], dtype=torch.float64)
        spot_direction = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
        intensity = torch.tensor([100.0], dtype=torch.float64)
        inner_angle = torch.tensor([math.radians(15)], dtype=torch.float64)
        outer_angle = torch.tensor([math.radians(30)], dtype=torch.float64)

        irradiance, _ = spotlight(
            light_pos, surface_pos, spot_direction,
            intensity=intensity, inner_angle=inner_angle, outer_angle=outer_angle
        )

        assert irradiance.item() == pytest.approx(0.0, abs=1e-6)

    def test_light_direction_normalized(self):
        """Returned light direction is normalized."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.tensor([[0.0, 5.0, 0.0]])
        surface_pos = torch.tensor([[3.0, 2.0, 4.0]])
        spot_direction = torch.tensor([[0.0, -1.0, 0.0]])
        intensity = torch.tensor([100.0])
        inner_angle = torch.tensor([math.radians(45)])
        outer_angle = torch.tensor([math.radians(60)])

        _, light_dir = spotlight(
            light_pos, surface_pos, spot_direction,
            intensity=intensity, inner_angle=inner_angle, outer_angle=outer_angle
        )

        norm = light_dir.norm(dim=-1)
        torch.testing.assert_close(norm, torch.ones_like(norm), rtol=1e-5, atol=1e-7)


class TestSpotlightGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Passes gradcheck for basic inputs."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64, requires_grad=True)
        surface_pos = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
        spot_direction = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64, requires_grad=True)
        intensity = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)
        inner_angle = torch.tensor([math.radians(30)], dtype=torch.float64, requires_grad=True)
        outer_angle = torch.tensor([math.radians(45)], dtype=torch.float64, requires_grad=True)

        def func(lp, sp, sd, i, ia, oa):
            irr, _ = spotlight(lp, sp, sd, intensity=i, inner_angle=ia, outer_angle=oa)
            return irr

        assert gradcheck(
            func,
            (light_pos, surface_pos, spot_direction, intensity, inner_angle, outer_angle),
            raise_exception=True
        )


class TestSpotlightDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.randn(5, 3, dtype=torch.float32)
        surface_pos = torch.randn(5, 3, dtype=torch.float32)
        spot_direction = torch.randn(5, 3, dtype=torch.float32)
        spot_direction = spot_direction / spot_direction.norm(dim=-1, keepdim=True)
        intensity = torch.full((5,), 100.0, dtype=torch.float32)
        inner_angle = torch.full((5,), 0.5, dtype=torch.float32)
        outer_angle = torch.full((5,), 0.8, dtype=torch.float32)

        irradiance, light_dir = spotlight(
            light_pos, surface_pos, spot_direction,
            intensity=intensity, inner_angle=inner_angle, outer_angle=outer_angle
        )

        assert irradiance.dtype == torch.float32
        assert light_dir.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.graphics.lighting import spotlight

        light_pos = torch.randn(5, 3, dtype=torch.float64)
        surface_pos = torch.randn(5, 3, dtype=torch.float64)
        spot_direction = torch.randn(5, 3, dtype=torch.float64)
        spot_direction = spot_direction / spot_direction.norm(dim=-1, keepdim=True)
        intensity = torch.full((5,), 100.0, dtype=torch.float64)
        inner_angle = torch.full((5,), 0.5, dtype=torch.float64)
        outer_angle = torch.full((5,), 0.8, dtype=torch.float64)

        irradiance, light_dir = spotlight(
            light_pos, surface_pos, spot_direction,
            intensity=intensity, inner_angle=inner_angle, outer_angle=outer_angle
        )

        assert irradiance.dtype == torch.float64
        assert light_dir.dtype == torch.float64
```

### Step 2.2: Run test to verify it fails

Run: `uv run pytest tests/torchscience/graphics/lighting/test__spotlight.py -v`
Expected: FAIL with "cannot import name 'spotlight'"

### Step 2.3-2.12: Implement following phong pattern

Follow the same pattern as phong (Steps 1.3-1.12) with:

**Schema:**
```cpp
module.def("spotlight(Tensor light_pos, Tensor surface_pos, Tensor spot_direction, Tensor intensity, Tensor inner_angle, Tensor outer_angle) -> (Tensor, Tensor)");
module.def("spotlight_backward(Tensor grad_irradiance, Tensor grad_light_dir, Tensor light_pos, Tensor surface_pos, Tensor spot_direction, Tensor intensity, Tensor inner_angle, Tensor outer_angle) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
```

**Commit:**
```bash
git commit -m "feat(lighting): add spotlight operator"
```

---

## Task 3: Add `reinhard` tone mapping operator

**Goal:** Implement Reinhard tone mapping for HDR to LDR conversion.

**Formula:**
```
Basic:    L_out = L_in / (1 + L_in)
Extended: L_out = L_in * (1 + L_in/L_white^2) / (1 + L_in)
```

**Files:**
- Create: `tests/torchscience/graphics/tone_mapping/test__reinhard.py`
- Create: `src/torchscience/csrc/kernel/graphics/tone_mapping/reinhard.h`
- Create: `src/torchscience/csrc/kernel/graphics/tone_mapping/reinhard_backward.h`
- Create: `src/torchscience/csrc/cpu/graphics/tone_mapping/reinhard.h`
- Create: `src/torchscience/csrc/meta/graphics/tone_mapping/reinhard.h`
- Create: `src/torchscience/csrc/autograd/graphics/tone_mapping/reinhard.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/graphics/tone_mapping/__init__.py`
- Create: `src/torchscience/graphics/tone_mapping/_reinhard.py`
- Modify: `src/torchscience/graphics/__init__.py`

### Step 3.1: Write the failing test

Create `tests/torchscience/graphics/tone_mapping/__init__.py` (empty file).

Create `tests/torchscience/graphics/tone_mapping/test__reinhard.py`:

```python
"""Tests for Reinhard tone mapping operator."""

import pytest
import torch
from torch.autograd import gradcheck


class TestReinhardBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_matches_input(self):
        """Output shape matches input shape."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(10, 3) * 10.0  # HDR values

        result = reinhard(input_hdr)

        assert result.shape == input_hdr.shape

    def test_output_range_basic(self):
        """Basic reinhard maps to [0, 1) for non-negative input."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(100, 3) * 100.0  # Large HDR values

        result = reinhard(input_hdr)

        assert (result >= 0).all()
        assert (result < 1).all()

    def test_output_range_extended(self):
        """Extended reinhard with white_point maps to [0, 1]."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(100, 3) * 10.0
        white_point = torch.tensor(10.0)

        result = reinhard(input_hdr, white_point=white_point)

        assert (result >= 0).all()
        assert (result <= 1.0 + 1e-6).all()


class TestReinhardCorrectness:
    """Tests for numerical correctness."""

    def test_basic_formula(self):
        """Basic reinhard: L_out = L_in / (1 + L_in)."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.tensor([0.0, 1.0, 2.0, 10.0], dtype=torch.float64)

        result = reinhard(input_hdr)

        expected = input_hdr / (1 + input_hdr)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)

    def test_extended_formula(self):
        """Extended reinhard: L_out = L_in * (1 + L_in/L_w^2) / (1 + L_in)."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
        white_point = torch.tensor(4.0, dtype=torch.float64)

        result = reinhard(input_hdr, white_point=white_point)

        L_w_sq = white_point ** 2
        expected = input_hdr * (1 + input_hdr / L_w_sq) / (1 + input_hdr)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)

    def test_white_point_maps_to_one(self):
        """Input at white_point maps to 1.0."""
        from torchscience.graphics.tone_mapping import reinhard

        white_point = torch.tensor(8.0, dtype=torch.float64)
        input_hdr = white_point.clone()

        result = reinhard(input_hdr.unsqueeze(0), white_point=white_point)

        torch.testing.assert_close(
            result,
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-5, atol=1e-7
        )

    def test_zero_input(self):
        """Zero input maps to zero."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.zeros(5, dtype=torch.float64)

        result = reinhard(input_hdr)

        torch.testing.assert_close(result, torch.zeros(5, dtype=torch.float64))


class TestReinhardGradients:
    """Tests for gradient computation."""

    def test_gradcheck_basic(self):
        """Passes gradcheck for basic reinhard."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(5, dtype=torch.float64, requires_grad=True) * 5.0 + 0.1

        def func(x):
            return reinhard(x)

        assert gradcheck(func, (input_hdr,), raise_exception=True)

    def test_gradcheck_extended(self):
        """Passes gradcheck for extended reinhard with white_point."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(5, dtype=torch.float64, requires_grad=True) * 5.0 + 0.1
        white_point = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)

        def func(x, wp):
            return reinhard(x, white_point=wp)

        assert gradcheck(func, (input_hdr, white_point), raise_exception=True)


class TestReinhardDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(10, dtype=torch.float32) * 10.0

        result = reinhard(input_hdr)

        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(10, dtype=torch.float64) * 10.0

        result = reinhard(input_hdr)

        assert result.dtype == torch.float64
```

### Step 3.2: Run test to verify it fails

Run: `uv run pytest tests/torchscience/graphics/tone_mapping/test__reinhard.py -v`
Expected: FAIL with "cannot import name 'reinhard'"

### Step 3.3-3.12: Implement following phong pattern

**Schema:**
```cpp
module.def("reinhard(Tensor input, Tensor? white_point) -> Tensor");
module.def("reinhard_backward(Tensor grad_output, Tensor input, Tensor? white_point) -> (Tensor, Tensor)");
```

**Commit:**
```bash
git commit -m "feat(tone_mapping): add reinhard operator"
```

---

## Task 4: Add `cube_mapping` operator

**Goal:** Convert 3D direction to cubemap face index and UV coordinates.

**Algorithm:**
```
1. Find axis with largest absolute component
2. Determine face (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z)
3. Compute UV from remaining components
```

**Files:**
- Create: `tests/torchscience/graphics/texture_mapping/test__cube_mapping.py`
- Create: `src/torchscience/csrc/kernel/graphics/texture_mapping/cube_mapping.h`
- Create: `src/torchscience/csrc/cpu/graphics/texture_mapping/cube_mapping.h`
- Create: `src/torchscience/csrc/meta/graphics/texture_mapping/cube_mapping.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/graphics/texture_mapping/__init__.py`
- Create: `src/torchscience/graphics/texture_mapping/_cube_mapping.py`
- Modify: `src/torchscience/graphics/__init__.py`

### Step 4.1: Write the failing test

Create `tests/torchscience/graphics/texture_mapping/__init__.py` (empty file).

Create `tests/torchscience/graphics/texture_mapping/test__cube_mapping.py`:

```python
"""Tests for cube_mapping texture mapping operator."""

import pytest
import torch


class TestCubeMappingBasic:
    """Tests for basic shape and property verification."""

    def test_output_shapes(self):
        """Output shapes are correct."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.randn(10, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        face, u, v = cube_mapping(direction)

        assert face.shape == (10,)
        assert u.shape == (10,)
        assert v.shape == (10,)

    def test_face_indices_valid(self):
        """Face indices are in range [0, 5]."""
        from torchscience.graphics.texture_mapping import cube_mapping

        torch.manual_seed(42)
        direction = torch.randn(100, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        face, _, _ = cube_mapping(direction)

        assert (face >= 0).all()
        assert (face <= 5).all()

    def test_uv_range(self):
        """UV coordinates are in range [0, 1]."""
        from torchscience.graphics.texture_mapping import cube_mapping

        torch.manual_seed(42)
        direction = torch.randn(100, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        _, u, v = cube_mapping(direction)

        assert (u >= 0).all()
        assert (u <= 1).all()
        assert (v >= 0).all()
        assert (v <= 1).all()


class TestCubeMappingCorrectness:
    """Tests for numerical correctness."""

    def test_positive_x_axis(self):
        """Direction along +X maps to face 0."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[1.0, 0.0, 0.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 0  # +X face
        torch.testing.assert_close(u, torch.tensor([0.5]), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(v, torch.tensor([0.5]), rtol=1e-5, atol=1e-7)

    def test_negative_x_axis(self):
        """Direction along -X maps to face 1."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[-1.0, 0.0, 0.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 1  # -X face

    def test_positive_y_axis(self):
        """Direction along +Y maps to face 2."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[0.0, 1.0, 0.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 2  # +Y face

    def test_negative_y_axis(self):
        """Direction along -Y maps to face 3."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[0.0, -1.0, 0.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 3  # -Y face

    def test_positive_z_axis(self):
        """Direction along +Z maps to face 4."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[0.0, 0.0, 1.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 4  # +Z face

    def test_negative_z_axis(self):
        """Direction along -Z maps to face 5."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[0.0, 0.0, -1.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 5  # -Z face


class TestCubeMappingDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.randn(5, 3, dtype=torch.float32)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        face, u, v = cube_mapping(direction)

        assert face.dtype == torch.int64  # Face index is int
        assert u.dtype == torch.float32
        assert v.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.randn(5, 3, dtype=torch.float64)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        face, u, v = cube_mapping(direction)

        assert face.dtype == torch.int64
        assert u.dtype == torch.float64
        assert v.dtype == torch.float64
```

### Step 4.2: Run test to verify it fails

Run: `uv run pytest tests/torchscience/graphics/texture_mapping/test__cube_mapping.py -v`
Expected: FAIL with "cannot import name 'cube_mapping'"

### Step 4.3-4.12: Implement following phong pattern

Note: cube_mapping is non-differentiable (returns discrete face indices), so no autograd wrapper needed.

**Schema:**
```cpp
module.def("cube_mapping(Tensor direction) -> (Tensor, Tensor, Tensor)");
```

**Commit:**
```bash
git commit -m "feat(texture_mapping): add cube_mapping operator"
```

---

## Task 5: Add `perspective_projection` operator

**Goal:** Generate a perspective projection matrix from camera parameters.

**Matrix:**
```
| f/aspect  0    0              0           |
| 0         f    0              0           |
| 0         0    (f+n)/(n-f)    2fn/(n-f)   |
| 0         0    -1             0           |

where f = 1/tan(fov/2)
```

**Files:**
- Create: `tests/torchscience/graphics/projection/test__perspective_projection.py`
- Create: `src/torchscience/csrc/kernel/graphics/projection/perspective_projection.h`
- Create: `src/torchscience/csrc/kernel/graphics/projection/perspective_projection_backward.h`
- Create: `src/torchscience/csrc/cpu/graphics/projection/perspective_projection.h`
- Create: `src/torchscience/csrc/meta/graphics/projection/perspective_projection.h`
- Create: `src/torchscience/csrc/autograd/graphics/projection/perspective_projection.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/graphics/projection/__init__.py`
- Create: `src/torchscience/graphics/projection/_perspective_projection.py`
- Modify: `src/torchscience/graphics/__init__.py`

### Step 5.1: Write the failing test

Create `tests/torchscience/graphics/projection/__init__.py` (empty file).

Create `tests/torchscience/graphics/projection/test__perspective_projection.py`:

```python
"""Tests for perspective projection matrix generation."""

import math
import pytest
import torch
from torch.autograd import gradcheck


class TestPerspectiveProjectionBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_single(self):
        """Output shape is (4, 4) for single sample."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.radians(60))
        aspect = torch.tensor(16.0 / 9.0)
        near = torch.tensor(0.1)
        far = torch.tensor(100.0)

        result = perspective_projection(fov, aspect, near, far)

        assert result.shape == (4, 4)

    def test_output_shape_batch(self):
        """Output shape is (batch, 4, 4) for batched input."""
        from torchscience.graphics.projection import perspective_projection

        batch = 5
        fov = torch.full((batch,), math.radians(60))
        aspect = torch.full((batch,), 16.0 / 9.0)
        near = torch.full((batch,), 0.1)
        far = torch.full((batch,), 100.0)

        result = perspective_projection(fov, aspect, near, far)

        assert result.shape == (batch, 4, 4)


class TestPerspectiveProjectionCorrectness:
    """Tests for numerical correctness."""

    def test_matrix_structure(self):
        """Matrix has correct zero/non-zero pattern."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.radians(60), dtype=torch.float64)
        aspect = torch.tensor(16.0 / 9.0, dtype=torch.float64)
        near = torch.tensor(0.1, dtype=torch.float64)
        far = torch.tensor(100.0, dtype=torch.float64)

        result = perspective_projection(fov, aspect, near, far)

        # Check zero elements
        assert result[0, 1].item() == 0
        assert result[0, 2].item() == 0
        assert result[0, 3].item() == 0
        assert result[1, 0].item() == 0
        assert result[1, 2].item() == 0
        assert result[1, 3].item() == 0
        assert result[2, 0].item() == 0
        assert result[2, 1].item() == 0
        assert result[3, 0].item() == 0
        assert result[3, 1].item() == 0
        assert result[3, 3].item() == 0

        # Check -1 at (3, 2)
        torch.testing.assert_close(
            result[3, 2],
            torch.tensor(-1.0, dtype=torch.float64),
            rtol=1e-5, atol=1e-7
        )

    def test_formula_values(self):
        """Matrix values match the formula."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.radians(90), dtype=torch.float64)  # 90 degrees
        aspect = torch.tensor(1.0, dtype=torch.float64)  # Square
        near = torch.tensor(1.0, dtype=torch.float64)
        far = torch.tensor(10.0, dtype=torch.float64)

        result = perspective_projection(fov, aspect, near, far)

        # f = 1/tan(45) = 1
        f = 1.0 / math.tan(math.radians(45))
        expected_00 = f / 1.0  # f/aspect = 1
        expected_11 = f  # 1
        expected_22 = (far + near) / (near - far)  # (10+1)/(1-10) = -11/9
        expected_23 = (2 * far * near) / (near - far)  # 20/(1-10) = -20/9

        torch.testing.assert_close(result[0, 0], torch.tensor(expected_00, dtype=torch.float64), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(result[1, 1], torch.tensor(expected_11, dtype=torch.float64), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(result[2, 2], torch.tensor(expected_22, dtype=torch.float64), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(result[2, 3], torch.tensor(expected_23, dtype=torch.float64), rtol=1e-4, atol=1e-6)

    def test_near_plane_maps_to_minus_one(self):
        """Point at near plane maps to z=-1 in NDC."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.radians(60), dtype=torch.float64)
        aspect = torch.tensor(1.0, dtype=torch.float64)
        near = torch.tensor(0.5, dtype=torch.float64)
        far = torch.tensor(100.0, dtype=torch.float64)

        P = perspective_projection(fov, aspect, near, far)

        # Point at near plane: (0, 0, -near, 1)
        point = torch.tensor([0.0, 0.0, -0.5, 1.0], dtype=torch.float64)
        clip = P @ point
        ndc_z = clip[2] / clip[3]

        torch.testing.assert_close(ndc_z, torch.tensor(-1.0, dtype=torch.float64), rtol=1e-4, atol=1e-6)

    def test_far_plane_maps_to_plus_one(self):
        """Point at far plane maps to z=+1 in NDC."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.radians(60), dtype=torch.float64)
        aspect = torch.tensor(1.0, dtype=torch.float64)
        near = torch.tensor(0.5, dtype=torch.float64)
        far = torch.tensor(100.0, dtype=torch.float64)

        P = perspective_projection(fov, aspect, near, far)

        # Point at far plane: (0, 0, -far, 1)
        point = torch.tensor([0.0, 0.0, -100.0, 1.0], dtype=torch.float64)
        clip = P @ point
        ndc_z = clip[2] / clip[3]

        torch.testing.assert_close(ndc_z, torch.tensor(1.0, dtype=torch.float64), rtol=1e-4, atol=1e-6)


class TestPerspectiveProjectionGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Passes gradcheck."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(math.radians(60), dtype=torch.float64, requires_grad=True)
        aspect = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        near = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        far = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)

        def func(f, a, n, fa):
            return perspective_projection(f, a, n, fa)

        assert gradcheck(func, (fov, aspect, near, far), raise_exception=True)


class TestPerspectiveProjectionDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(1.0, dtype=torch.float32)
        aspect = torch.tensor(1.5, dtype=torch.float32)
        near = torch.tensor(0.1, dtype=torch.float32)
        far = torch.tensor(100.0, dtype=torch.float32)

        result = perspective_projection(fov, aspect, near, far)

        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.graphics.projection import perspective_projection

        fov = torch.tensor(1.0, dtype=torch.float64)
        aspect = torch.tensor(1.5, dtype=torch.float64)
        near = torch.tensor(0.1, dtype=torch.float64)
        far = torch.tensor(100.0, dtype=torch.float64)

        result = perspective_projection(fov, aspect, near, far)

        assert result.dtype == torch.float64
```

### Step 5.2: Run test to verify it fails

Run: `uv run pytest tests/torchscience/graphics/projection/test__perspective_projection.py -v`
Expected: FAIL with "cannot import name 'perspective_projection'"

### Step 5.3-5.12: Implement following phong pattern

**Schema:**
```cpp
module.def("perspective_projection(Tensor fov, Tensor aspect, Tensor near, Tensor far) -> Tensor");
module.def("perspective_projection_backward(Tensor grad_output, Tensor fov, Tensor aspect, Tensor near, Tensor far) -> (Tensor, Tensor, Tensor, Tensor)");
```

**Commit:**
```bash
git commit -m "feat(projection): add perspective_projection operator"
```

---

## Task 6: Update graphics __init__.py

Modify `src/torchscience/graphics/__init__.py`:

```python
from . import color, lighting, projection, shading, texture_mapping, tone_mapping

__all__ = [
    "color",
    "lighting",
    "projection",
    "shading",
    "texture_mapping",
    "tone_mapping",
]
```

### Step 6.1: Commit

```bash
git add src/torchscience/graphics/__init__.py
git commit -m "feat(graphics): export all submodules"
```

---

## Task 7: Run full test suite

Run: `uv run pytest tests/torchscience/graphics/ -v`
Expected: All tests PASS

---

## Task 8: Final commit

```bash
git add -A
git commit -m "feat(graphics): complete MVP with 5 new operators"
```

---

## Status: DRAFT - Ready for refinement
