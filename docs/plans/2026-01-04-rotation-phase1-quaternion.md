# Phase 1: Core Quaternion Operations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the Quaternion tensorclass and core quaternion operations: multiply, inverse, normalize, apply, to_matrix, from_matrix, slerp.

**Architecture:** Each operation follows torchscience's layered pattern: Python API → torch.ops → PyTorch Dispatcher → Backend implementations (CPU, Meta, Autograd). Quaternions use scalar-first (wxyz) convention. Operations are standalone functions that accept/return Quaternion tensorclass instances.

**Tech Stack:** PyTorch, tensordict tensorclass, C++17 kernels

---

## Task 1: Quaternion Tensorclass

**Files:**
- Create: `src/torchscience/geometry/transform/_quaternion.py`
- Modify: `src/torchscience/geometry/transform/__init__.py`
- Test: `tests/torchscience/geometry/transform/test__quaternion.py`

**Step 1: Write the failing test**

Create `tests/torchscience/geometry/transform/test__quaternion.py`:

```python
"""Tests for Quaternion tensorclass."""

import pytest
import torch

from torchscience.geometry.transform import Quaternion, quaternion


class TestQuaternionConstruction:
    """Tests for Quaternion construction."""

    def test_from_tensor(self):
        """Create Quaternion from tensor."""
        wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q = Quaternion(wxyz=wxyz)
        assert q.wxyz.shape == (4,)
        assert torch.allclose(q.wxyz, wxyz)

    def test_batch(self):
        """Batch of quaternions."""
        wxyz = torch.randn(10, 4)
        q = Quaternion(wxyz=wxyz)
        assert q.wxyz.shape == (10, 4)

    def test_factory_function(self):
        """Create via quaternion() factory."""
        wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q = quaternion(wxyz)
        assert isinstance(q, Quaternion)
        assert torch.allclose(q.wxyz, wxyz)

    def test_invalid_shape(self):
        """Raise error for wrong last dimension."""
        with pytest.raises(ValueError, match="last dimension 4"):
            quaternion(torch.randn(3))
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body && uv run pytest tests/torchscience/geometry/transform/test__quaternion.py -v`

Expected: FAIL with "cannot import name 'Quaternion'"

**Step 3: Write minimal implementation**

Create `src/torchscience/geometry/transform/_quaternion.py`:

```python
"""Quaternion representation and operations."""

from __future__ import annotations

from tensordict.tensorclass import tensorclass
from torch import Tensor


@tensorclass
class Quaternion:
    """Unit quaternion representing a 3D rotation.

    Uses scalar-first (wxyz) convention: q = w + xi + yj + zk.

    Attributes
    ----------
    wxyz : Tensor
        Quaternion components in [w, x, y, z] order, shape (..., 4).
        For unit quaternions: w^2 + x^2 + y^2 + z^2 = 1.

    Examples
    --------
    Identity rotation:
        Quaternion(wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]))

    90-degree rotation around z-axis:
        Quaternion(wxyz=torch.tensor([0.7071, 0.0, 0.0, 0.7071]))

    Batch of quaternions:
        Quaternion(wxyz=torch.randn(100, 4))
    """

    wxyz: Tensor


def quaternion(wxyz: Tensor) -> Quaternion:
    """Create quaternion from wxyz tensor.

    Parameters
    ----------
    wxyz : Tensor
        Quaternion components [w, x, y, z], shape (..., 4).

    Returns
    -------
    Quaternion
        Quaternion instance.

    Raises
    ------
    ValueError
        If wxyz does not have last dimension 4.

    Examples
    --------
    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> q.wxyz
    tensor([1., 0., 0., 0.])
    """
    if wxyz.shape[-1] != 4:
        raise ValueError(
            f"quaternion: wxyz must have last dimension 4, got {wxyz.shape[-1]}"
        )
    return Quaternion(wxyz=wxyz)
```

**Step 4: Update __init__.py**

Modify `src/torchscience/geometry/transform/__init__.py`:

```python
"""Geometry transform operations."""

from torchscience.geometry.transform._quaternion import Quaternion, quaternion
from torchscience.geometry.transform._reflect import reflect
from torchscience.geometry.transform._refract import refract

__all__ = [
    "Quaternion",
    "quaternion",
    "reflect",
    "refract",
]
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body && uv run pytest tests/torchscience/geometry/transform/test__quaternion.py -v`

Expected: PASS (4 tests)

**Step 6: Commit**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body add src/torchscience/geometry/transform/_quaternion.py src/torchscience/geometry/transform/__init__.py tests/torchscience/geometry/transform/test__quaternion.py
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body commit -m "feat(geometry.transform): add Quaternion tensorclass"
```

---

## Task 2: quaternion_multiply - Schema and Kernel

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp` (add schema)
- Create: `src/torchscience/csrc/kernel/geometry/transform/quaternion_multiply.h`
- Create: `src/torchscience/csrc/kernel/geometry/transform/quaternion_multiply_backward.h`

**Step 1: Add schema to torchscience.cpp**

Add after the reflect/refract schemas (around line 423):

```cpp
  // Quaternion operations
  module.def("quaternion_multiply(Tensor q1, Tensor q2) -> Tensor");
  module.def("quaternion_multiply_backward(Tensor grad_output, Tensor q1, Tensor q2) -> (Tensor, Tensor)");
```

**Step 2: Create forward kernel**

Create `src/torchscience/csrc/kernel/geometry/transform/quaternion_multiply.h`:

```cpp
#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Hamilton product of two quaternions (scalar-first wxyz convention).
 * q1 * q2 represents rotation q1 followed by rotation q2.
 */
template <typename T>
void quaternion_multiply_scalar(const T* q1, const T* q2, T* output) {
  const T w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
  const T w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];

  output[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
  output[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
  output[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
  output[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
}

}  // namespace torchscience::kernel::geometry::transform
```

**Step 3: Create backward kernel**

Create `src/torchscience/csrc/kernel/geometry/transform/quaternion_multiply_backward.h`:

```cpp
#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Backward pass for quaternion multiplication.
 * Computes gradients w.r.t. q1 and q2 given gradient of output.
 */
template <typename T>
void quaternion_multiply_backward_scalar(
    const T* grad_output,
    const T* q1,
    const T* q2,
    T* grad_q1,
    T* grad_q2
) {
  const T w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
  const T w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
  const T gw = grad_output[0], gx = grad_output[1];
  const T gy = grad_output[2], gz = grad_output[3];

  // Gradient w.r.t. q1
  // d(output)/d(w1) = [w2, x2, y2, z2]
  // d(output)/d(x1) = [-x2, w2, -z2, y2]
  // d(output)/d(y1) = [-y2, z2, w2, -x2]
  // d(output)/d(z1) = [-z2, -y2, x2, w2]
  grad_q1[0] = gw * w2 + gx * x2 + gy * y2 + gz * z2;
  grad_q1[1] = -gw * x2 + gx * w2 + gy * z2 - gz * y2;
  grad_q1[2] = -gw * y2 - gx * z2 + gy * w2 + gz * x2;
  grad_q1[3] = -gw * z2 + gx * y2 - gy * x2 + gz * w2;

  // Gradient w.r.t. q2
  // d(output)/d(w2) = [w1, x1, y1, z1]
  // d(output)/d(x2) = [-x1, w1, z1, -y1]
  // d(output)/d(y2) = [-y1, -z1, w1, x1]
  // d(output)/d(z2) = [-z1, y1, -x1, w1]
  grad_q2[0] = gw * w1 + gx * x1 + gy * y1 + gz * z1;
  grad_q2[1] = -gw * x1 + gx * w1 - gy * z1 + gz * y1;
  grad_q2[2] = -gw * y1 + gx * z1 + gy * w1 - gz * x1;
  grad_q2[3] = -gw * z1 - gx * y1 + gy * x1 + gz * w1;
}

}  // namespace torchscience::kernel::geometry::transform
```

**Step 4: Commit**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body add src/torchscience/csrc/torchscience.cpp src/torchscience/csrc/kernel/geometry/transform/quaternion_multiply.h src/torchscience/csrc/kernel/geometry/transform/quaternion_multiply_backward.h
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body commit -m "feat(geometry.transform): add quaternion_multiply kernels and schema"
```

---

## Task 3: quaternion_multiply - CPU Backend

**Files:**
- Create: `src/torchscience/csrc/cpu/geometry/transform/quaternion_multiply.h`
- Modify: `src/torchscience/csrc/torchscience.cpp` (add include)

**Step 1: Create CPU implementation**

Create `src/torchscience/csrc/cpu/geometry/transform/quaternion_multiply.h`:

```cpp
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/transform/quaternion_multiply.h"
#include "../../../kernel/geometry/transform/quaternion_multiply_backward.h"

namespace torchscience::cpu::geometry::transform {

inline at::Tensor quaternion_multiply(const at::Tensor& q1, const at::Tensor& q2) {
  TORCH_CHECK(q1.size(-1) == 4, "quaternion_multiply: q1 must have last dimension 4, got ", q1.size(-1));
  TORCH_CHECK(q2.size(-1) == 4, "quaternion_multiply: q2 must have last dimension 4, got ", q2.size(-1));
  TORCH_CHECK(q1.scalar_type() == q2.scalar_type(),
              "quaternion_multiply: q1 and q2 must have the same dtype");

  // Broadcast batch dimensions
  auto batch1 = q1.sizes().slice(0, q1.dim() - 1);
  auto batch2 = q2.sizes().slice(0, q2.dim() - 1);
  auto broadcast_shape = at::infer_size(batch1, batch2);

  // Expand inputs to broadcast shape
  std::vector<int64_t> full_shape1(broadcast_shape.begin(), broadcast_shape.end());
  full_shape1.push_back(4);
  std::vector<int64_t> full_shape2(broadcast_shape.begin(), broadcast_shape.end());
  full_shape2.push_back(4);

  auto q1_expanded = q1.expand(full_shape1).contiguous();
  auto q2_expanded = q2.expand(full_shape2).contiguous();

  auto output = at::empty_like(q1_expanded);

  const int64_t num_quats = q1_expanded.numel() / 4;

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    q1.scalar_type(),
    "quaternion_multiply_cpu",
    [&] {
      const scalar_t* q1_ptr = q1_expanded.data_ptr<scalar_t>();
      const scalar_t* q2_ptr = q2_expanded.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      at::parallel_for(0, num_quats, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_multiply_scalar(
            q1_ptr + i * 4,
            q2_ptr + i * 4,
            output_ptr + i * 4
          );
        }
      });
    }
  );

  return output;
}

inline std::tuple<at::Tensor, at::Tensor> quaternion_multiply_backward(
    const at::Tensor& grad_output,
    const at::Tensor& q1,
    const at::Tensor& q2
) {
  TORCH_CHECK(grad_output.size(-1) == 4, "quaternion_multiply_backward: grad_output must have last dimension 4");
  TORCH_CHECK(q1.size(-1) == 4, "quaternion_multiply_backward: q1 must have last dimension 4");
  TORCH_CHECK(q2.size(-1) == 4, "quaternion_multiply_backward: q2 must have last dimension 4");

  auto grad_q1 = at::empty_like(q1);
  auto grad_q2 = at::empty_like(q2);

  const int64_t num_quats = q1.numel() / 4;

  auto grad_output_contig = grad_output.contiguous();
  auto q1_contig = q1.contiguous();
  auto q2_contig = q2.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    q1.scalar_type(),
    "quaternion_multiply_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
      const scalar_t* q1_ptr = q1_contig.data_ptr<scalar_t>();
      const scalar_t* q2_ptr = q2_contig.data_ptr<scalar_t>();
      scalar_t* grad_q1_ptr = grad_q1.data_ptr<scalar_t>();
      scalar_t* grad_q2_ptr = grad_q2.data_ptr<scalar_t>();

      at::parallel_for(0, num_quats, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_multiply_backward_scalar(
            grad_output_ptr + i * 4,
            q1_ptr + i * 4,
            q2_ptr + i * 4,
            grad_q1_ptr + i * 4,
            grad_q2_ptr + i * 4
          );
        }
      });
    }
  );

  return std::make_tuple(grad_q1, grad_q2);
}

}  // namespace torchscience::cpu::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("quaternion_multiply", &torchscience::cpu::geometry::transform::quaternion_multiply);
  m.impl("quaternion_multiply_backward", &torchscience::cpu::geometry::transform::quaternion_multiply_backward);
}
```

**Step 2: Add include to torchscience.cpp**

Add after the reflect.h include (around line 73):

```cpp
#include "cpu/geometry/transform/quaternion_multiply.h"
```

**Step 3: Commit**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body add src/torchscience/csrc/cpu/geometry/transform/quaternion_multiply.h src/torchscience/csrc/torchscience.cpp
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body commit -m "feat(geometry.transform): add quaternion_multiply CPU backend"
```

---

## Task 4: quaternion_multiply - Meta Backend

**Files:**
- Create: `src/torchscience/csrc/meta/geometry/transform/quaternion_multiply.h`
- Modify: `src/torchscience/csrc/torchscience.cpp` (add include)

**Step 1: Create Meta implementation**

Create `src/torchscience/csrc/meta/geometry/transform/quaternion_multiply.h`:

```cpp
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::transform {

inline at::Tensor quaternion_multiply(const at::Tensor& q1, const at::Tensor& q2) {
  TORCH_CHECK(q1.size(-1) == 4, "quaternion_multiply: q1 must have last dimension 4, got ", q1.size(-1));
  TORCH_CHECK(q2.size(-1) == 4, "quaternion_multiply: q2 must have last dimension 4, got ", q2.size(-1));

  // Infer broadcast shape for batch dimensions
  auto batch1 = q1.sizes().slice(0, q1.dim() - 1);
  auto batch2 = q2.sizes().slice(0, q2.dim() - 1);
  auto broadcast_shape = at::infer_size(batch1, batch2);

  std::vector<int64_t> output_shape(broadcast_shape.begin(), broadcast_shape.end());
  output_shape.push_back(4);

  return at::empty(output_shape, q1.options());
}

inline std::tuple<at::Tensor, at::Tensor> quaternion_multiply_backward(
    const at::Tensor& grad_output,
    const at::Tensor& q1,
    const at::Tensor& q2
) {
  return std::make_tuple(at::empty_like(q1), at::empty_like(q2));
}

}  // namespace torchscience::meta::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("quaternion_multiply", &torchscience::meta::geometry::transform::quaternion_multiply);
  m.impl("quaternion_multiply_backward", &torchscience::meta::geometry::transform::quaternion_multiply_backward);
}
```

**Step 2: Add include to torchscience.cpp**

Add after the meta reflect.h include (around line 129):

```cpp
#include "meta/geometry/transform/quaternion_multiply.h"
```

**Step 3: Commit**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body add src/torchscience/csrc/meta/geometry/transform/quaternion_multiply.h src/torchscience/csrc/torchscience.cpp
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body commit -m "feat(geometry.transform): add quaternion_multiply Meta backend"
```

---

## Task 5: quaternion_multiply - Autograd Backend

**Files:**
- Create: `src/torchscience/csrc/autograd/geometry/transform/quaternion_multiply.h`
- Modify: `src/torchscience/csrc/torchscience.cpp` (add include)

**Step 1: Create Autograd implementation**

Create `src/torchscience/csrc/autograd/geometry/transform/quaternion_multiply.h`:

```cpp
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::geometry::transform {

class QuaternionMultiplyFunction : public torch::autograd::Function<QuaternionMultiplyFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& q1,
      const at::Tensor& q2
  ) {
    ctx->save_for_backward({q1, q2});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::quaternion_multiply", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(q1, q2);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor q1 = saved[0];
    at::Tensor q2 = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::quaternion_multiply_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, q1, q2);

    return {std::get<0>(result), std::get<1>(result)};
  }
};

inline at::Tensor quaternion_multiply(const at::Tensor& q1, const at::Tensor& q2) {
  return QuaternionMultiplyFunction::apply(q1, q2);
}

}  // namespace torchscience::autograd::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("quaternion_multiply", &torchscience::autograd::geometry::transform::quaternion_multiply);
}
```

**Step 2: Add include to torchscience.cpp**

Add after the autograd reflect.h include (around line 97):

```cpp
#include "autograd/geometry/transform/quaternion_multiply.h"
```

**Step 3: Commit**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body add src/torchscience/csrc/autograd/geometry/transform/quaternion_multiply.h src/torchscience/csrc/torchscience.cpp
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body commit -m "feat(geometry.transform): add quaternion_multiply Autograd backend"
```

---

## Task 6: quaternion_multiply - Python API and Tests

**Files:**
- Modify: `src/torchscience/geometry/transform/_quaternion.py`
- Modify: `src/torchscience/geometry/transform/__init__.py`
- Modify: `tests/torchscience/geometry/transform/test__quaternion.py`

**Step 1: Add tests**

Append to `tests/torchscience/geometry/transform/test__quaternion.py`:

```python
from torchscience.geometry.transform import quaternion_multiply
from torch.autograd import gradcheck


class TestQuaternionMultiply:
    """Tests for quaternion_multiply."""

    def test_identity_left(self):
        """Multiplying by identity on left returns original."""
        identity = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        q = quaternion(torch.tensor([0.7071, 0.7071, 0.0, 0.0]))
        result = quaternion_multiply(identity, q)
        assert torch.allclose(result.wxyz, q.wxyz, atol=1e-5)

    def test_identity_right(self):
        """Multiplying by identity on right returns original."""
        identity = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        q = quaternion(torch.tensor([0.7071, 0.7071, 0.0, 0.0]))
        result = quaternion_multiply(q, identity)
        assert torch.allclose(result.wxyz, q.wxyz, atol=1e-5)

    def test_90_deg_rotations(self):
        """Two 90-degree rotations around z = 180-degree rotation."""
        # 90 degrees around z: [cos(45), 0, 0, sin(45)]
        q90z = quaternion(torch.tensor([0.7071067811865476, 0.0, 0.0, 0.7071067811865476]))
        result = quaternion_multiply(q90z, q90z)
        # 180 degrees around z: [0, 0, 0, 1]
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-5)

    def test_inverse_gives_identity(self):
        """q * q^(-1) = identity (conjugate for unit quaternion)."""
        q = quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        q_conj = quaternion(torch.tensor([0.5, -0.5, -0.5, -0.5]))
        result = quaternion_multiply(q, q_conj)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-5)

    def test_batch(self):
        """Batched multiplication."""
        q1 = quaternion(torch.randn(10, 4))
        q2 = quaternion(torch.randn(10, 4))
        result = quaternion_multiply(q1, q2)
        assert result.wxyz.shape == (10, 4)

    def test_broadcast(self):
        """Broadcasting batch dimensions."""
        q1 = quaternion(torch.randn(5, 1, 4))
        q2 = quaternion(torch.randn(1, 3, 4))
        result = quaternion_multiply(q1, q2)
        assert result.wxyz.shape == (5, 3, 4)

    def test_gradcheck(self):
        """Gradient check."""
        q1 = quaternion(torch.randn(5, 4, dtype=torch.float64, requires_grad=True))
        q2 = quaternion(torch.randn(5, 4, dtype=torch.float64, requires_grad=True))
        assert gradcheck(
            lambda a, b: quaternion_multiply(
                Quaternion(wxyz=a), Quaternion(wxyz=b)
            ).wxyz,
            (q1.wxyz, q2.wxyz),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradgradcheck(self):
        """Second-order gradient check."""
        from torch.autograd import gradgradcheck
        q1 = quaternion(torch.randn(3, 4, dtype=torch.float64, requires_grad=True))
        q2 = quaternion(torch.randn(3, 4, dtype=torch.float64, requires_grad=True))
        assert gradgradcheck(
            lambda a, b: quaternion_multiply(
                Quaternion(wxyz=a), Quaternion(wxyz=b)
            ).wxyz,
            (q1.wxyz, q2.wxyz),
            eps=1e-6,
            atol=1e-4,
        )
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body && uv run pytest tests/torchscience/geometry/transform/test__quaternion.py::TestQuaternionMultiply -v`

Expected: FAIL with "cannot import name 'quaternion_multiply'"

**Step 3: Add Python implementation**

Append to `src/torchscience/geometry/transform/_quaternion.py`:

```python
import torch

import torchscience._csrc  # noqa: F401 - Load C++ operators


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Multiply two quaternions (Hamilton product).

    Computes q1 * q2, representing rotation q1 followed by rotation q2.

    Parameters
    ----------
    q1 : Quaternion
        First quaternion, shape (..., 4).
    q2 : Quaternion
        Second quaternion, shape (..., 4). Batch dimensions broadcast with q1.

    Returns
    -------
    Quaternion
        Product q1 * q2, shape is broadcast of q1 and q2 batch dimensions.

    Examples
    --------
    >>> q1 = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # identity
    >>> q2 = quaternion(torch.tensor([0.7071, 0.7071, 0.0, 0.0]))  # 90deg around x
    >>> quaternion_multiply(q1, q2).wxyz
    tensor([0.7071, 0.7071, 0.0000, 0.0000])
    """
    result = torch.ops.torchscience.quaternion_multiply(q1.wxyz, q2.wxyz)
    return Quaternion(wxyz=result)
```

**Step 4: Update __init__.py**

Add `quaternion_multiply` to imports and `__all__` in `src/torchscience/geometry/transform/__init__.py`:

```python
from torchscience.geometry.transform._quaternion import (
    Quaternion,
    quaternion,
    quaternion_multiply,
)
```

And update `__all__`:
```python
__all__ = [
    "Quaternion",
    "quaternion",
    "quaternion_multiply",
    "reflect",
    "refract",
]
```

**Step 5: Rebuild and run tests**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body && uv sync && uv run pytest tests/torchscience/geometry/transform/test__quaternion.py -v`

Expected: PASS (all tests)

**Step 6: Commit**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body add src/torchscience/geometry/transform/_quaternion.py src/torchscience/geometry/transform/__init__.py tests/torchscience/geometry/transform/test__quaternion.py
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body commit -m "feat(geometry.transform): add quaternion_multiply Python API and tests"
```

---

## Task 7: quaternion_inverse

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp` (schema)
- Create: `src/torchscience/csrc/kernel/geometry/transform/quaternion_inverse.h`
- Create: `src/torchscience/csrc/kernel/geometry/transform/quaternion_inverse_backward.h`
- Create: `src/torchscience/csrc/cpu/geometry/transform/quaternion_inverse.h`
- Create: `src/torchscience/csrc/meta/geometry/transform/quaternion_inverse.h`
- Create: `src/torchscience/csrc/autograd/geometry/transform/quaternion_inverse.h`
- Modify: `src/torchscience/geometry/transform/_quaternion.py`
- Modify: `tests/torchscience/geometry/transform/test__quaternion.py`

**Step 1: Add schema**

Add to `src/torchscience/csrc/torchscience.cpp`:

```cpp
  module.def("quaternion_inverse(Tensor q) -> Tensor");
  module.def("quaternion_inverse_backward(Tensor grad_output, Tensor q) -> Tensor");
```

**Step 2: Create kernel**

Create `src/torchscience/csrc/kernel/geometry/transform/quaternion_inverse.h`:

```cpp
#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Inverse of a unit quaternion (conjugate).
 * For unit quaternion, inverse = conjugate = [w, -x, -y, -z].
 */
template <typename T>
void quaternion_inverse_scalar(const T* q, T* output) {
  output[0] = q[0];
  output[1] = -q[1];
  output[2] = -q[2];
  output[3] = -q[3];
}

}  // namespace torchscience::kernel::geometry::transform
```

Create `src/torchscience/csrc/kernel/geometry/transform/quaternion_inverse_backward.h`:

```cpp
#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Backward pass for quaternion inverse.
 * Since inverse is just conjugate: d(inverse)/dq = [1, -1, -1, -1]
 */
template <typename T>
void quaternion_inverse_backward_scalar(const T* grad_output, T* grad_q) {
  grad_q[0] = grad_output[0];
  grad_q[1] = -grad_output[1];
  grad_q[2] = -grad_output[2];
  grad_q[3] = -grad_output[3];
}

}  // namespace torchscience::kernel::geometry::transform
```

**Step 3: Create CPU backend**

Create `src/torchscience/csrc/cpu/geometry/transform/quaternion_inverse.h`:

```cpp
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/transform/quaternion_inverse.h"
#include "../../../kernel/geometry/transform/quaternion_inverse_backward.h"

namespace torchscience::cpu::geometry::transform {

inline at::Tensor quaternion_inverse(const at::Tensor& q) {
  TORCH_CHECK(q.size(-1) == 4, "quaternion_inverse: q must have last dimension 4, got ", q.size(-1));

  auto output = at::empty_like(q);
  const int64_t num_quats = q.numel() / 4;
  auto q_contig = q.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf, q.scalar_type(), "quaternion_inverse_cpu",
    [&] {
      const scalar_t* q_ptr = q_contig.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      at::parallel_for(0, num_quats, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_inverse_scalar(
            q_ptr + i * 4, output_ptr + i * 4
          );
        }
      });
    }
  );

  return output;
}

inline at::Tensor quaternion_inverse_backward(const at::Tensor& grad_output, const at::Tensor& q) {
  auto grad_q = at::empty_like(q);
  const int64_t num_quats = q.numel() / 4;
  auto grad_output_contig = grad_output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf, q.scalar_type(), "quaternion_inverse_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
      scalar_t* grad_q_ptr = grad_q.data_ptr<scalar_t>();

      at::parallel_for(0, num_quats, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_inverse_backward_scalar(
            grad_output_ptr + i * 4, grad_q_ptr + i * 4
          );
        }
      });
    }
  );

  return grad_q;
}

}  // namespace torchscience::cpu::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("quaternion_inverse", &torchscience::cpu::geometry::transform::quaternion_inverse);
  m.impl("quaternion_inverse_backward", &torchscience::cpu::geometry::transform::quaternion_inverse_backward);
}
```

**Step 4: Create Meta backend**

Create `src/torchscience/csrc/meta/geometry/transform/quaternion_inverse.h`:

```cpp
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::transform {

inline at::Tensor quaternion_inverse(const at::Tensor& q) {
  TORCH_CHECK(q.size(-1) == 4, "quaternion_inverse: q must have last dimension 4, got ", q.size(-1));
  return at::empty_like(q);
}

inline at::Tensor quaternion_inverse_backward(const at::Tensor& grad_output, const at::Tensor& q) {
  return at::empty_like(q);
}

}  // namespace torchscience::meta::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("quaternion_inverse", &torchscience::meta::geometry::transform::quaternion_inverse);
  m.impl("quaternion_inverse_backward", &torchscience::meta::geometry::transform::quaternion_inverse_backward);
}
```

**Step 5: Create Autograd backend**

Create `src/torchscience/csrc/autograd/geometry/transform/quaternion_inverse.h`:

```cpp
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::geometry::transform {

class QuaternionInverseFunction : public torch::autograd::Function<QuaternionInverseFunction> {
public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& q) {
    ctx->save_for_backward({q});
    at::AutoDispatchBelowAutograd guard;
    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::quaternion_inverse", "")
        .typed<at::Tensor(const at::Tensor&)>()
        .call(q);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor q = saved[0];
    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;
    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::quaternion_inverse_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, q);

    return {result};
  }
};

inline at::Tensor quaternion_inverse(const at::Tensor& q) {
  return QuaternionInverseFunction::apply(q);
}

}  // namespace torchscience::autograd::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("quaternion_inverse", &torchscience::autograd::geometry::transform::quaternion_inverse);
}
```

**Step 6: Add includes to torchscience.cpp**

Add the three includes to the appropriate sections.

**Step 7: Add Python API**

Add to `src/torchscience/geometry/transform/_quaternion.py`:

```python
def quaternion_inverse(q: Quaternion) -> Quaternion:
    """Compute inverse of a unit quaternion.

    For unit quaternions, the inverse equals the conjugate.

    Parameters
    ----------
    q : Quaternion
        Unit quaternion, shape (..., 4).

    Returns
    -------
    Quaternion
        Inverse quaternion q^(-1), shape (..., 4).

    Examples
    --------
    >>> q = quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    >>> q_inv = quaternion_inverse(q)
    >>> quaternion_multiply(q, q_inv).wxyz  # identity
    tensor([1., 0., 0., 0.])
    """
    result = torch.ops.torchscience.quaternion_inverse(q.wxyz)
    return Quaternion(wxyz=result)
```

**Step 8: Add tests**

Add to `tests/torchscience/geometry/transform/test__quaternion.py`:

```python
class TestQuaternionInverse:
    """Tests for quaternion_inverse."""

    def test_inverse_of_identity(self):
        """Inverse of identity is identity."""
        identity = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        result = quaternion_inverse(identity)
        assert torch.allclose(result.wxyz, identity.wxyz, atol=1e-5)

    def test_multiply_by_inverse_gives_identity(self):
        """q * q^(-1) = identity."""
        q = quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        q_inv = quaternion_inverse(q)
        result = quaternion_multiply(q, q_inv)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-5)

    def test_batch(self):
        """Batched inverse."""
        q = quaternion(torch.randn(10, 4))
        result = quaternion_inverse(q)
        assert result.wxyz.shape == (10, 4)

    def test_gradcheck(self):
        """Gradient check."""
        q = quaternion(torch.randn(5, 4, dtype=torch.float64, requires_grad=True))
        assert gradcheck(
            lambda x: quaternion_inverse(Quaternion(wxyz=x)).wxyz,
            (q.wxyz,),
            eps=1e-6,
            atol=1e-4,
        )
```

**Step 9: Update __init__.py and rebuild**

**Step 10: Commit**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body add -A
git -C /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/rotation-rigid-body commit -m "feat(geometry.transform): add quaternion_inverse"
```

---

## Task 8: quaternion_normalize

Follow the same pattern as Task 7. Key differences:

**Kernel** (`quaternion_normalize.h`):
```cpp
template <typename T>
void quaternion_normalize_scalar(const T* q, T* output) {
  const T norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  const T inv_norm = T(1) / norm;
  output[0] = q[0] * inv_norm;
  output[1] = q[1] * inv_norm;
  output[2] = q[2] * inv_norm;
  output[3] = q[3] * inv_norm;
}
```

**Backward kernel** (same as vector normalization):
```cpp
template <typename T>
void quaternion_normalize_backward_scalar(const T* grad_output, const T* q, T* grad_q) {
  const T norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  const T inv_norm = T(1) / norm;
  const T inv_norm3 = inv_norm * inv_norm * inv_norm;

  // d(q/||q||)/dq = (I - q*q^T/||q||^2) / ||q||
  const T dot = grad_output[0]*q[0] + grad_output[1]*q[1] +
                grad_output[2]*q[2] + grad_output[3]*q[3];

  grad_q[0] = (grad_output[0] - q[0] * dot * inv_norm * inv_norm) * inv_norm;
  grad_q[1] = (grad_output[1] - q[1] * dot * inv_norm * inv_norm) * inv_norm;
  grad_q[2] = (grad_output[2] - q[2] * dot * inv_norm * inv_norm) * inv_norm;
  grad_q[3] = (grad_output[3] - q[3] * dot * inv_norm * inv_norm) * inv_norm;
}
```

---

## Task 9: quaternion_apply

Rotate a 3D point by a quaternion using `q * p * q^(-1)` where p = [0, x, y, z].

**Kernel** (`quaternion_apply.h`):
```cpp
template <typename T>
void quaternion_apply_scalar(const T* q, const T* point, T* output) {
  const T w = q[0], x = q[1], y = q[2], z = q[3];
  const T px = point[0], py = point[1], pz = point[2];

  // Optimized quaternion rotation: v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
  // where q_xyz = [x, y, z]
  const T tx = T(2) * (y * pz - z * py);
  const T ty = T(2) * (z * px - x * pz);
  const T tz = T(2) * (x * py - y * px);

  output[0] = px + w * tx + (y * tz - z * ty);
  output[1] = py + w * ty + (z * tx - x * tz);
  output[2] = pz + w * tz + (x * ty - y * tx);
}
```

---

## Task 10: quaternion_to_matrix

Convert quaternion to 3x3 rotation matrix.

**Kernel** (`quaternion_to_matrix.h`):
```cpp
template <typename T>
void quaternion_to_matrix_scalar(const T* q, T* matrix) {
  const T w = q[0], x = q[1], y = q[2], z = q[3];

  const T x2 = x + x, y2 = y + y, z2 = z + z;
  const T xx = x * x2, xy = x * y2, xz = x * z2;
  const T yy = y * y2, yz = y * z2, zz = z * z2;
  const T wx = w * x2, wy = w * y2, wz = w * z2;

  // Row-major: matrix[row * 3 + col]
  matrix[0] = T(1) - (yy + zz);  matrix[1] = xy - wz;           matrix[2] = xz + wy;
  matrix[3] = xy + wz;           matrix[4] = T(1) - (xx + zz);  matrix[5] = yz - wx;
  matrix[6] = xz - wy;           matrix[7] = yz + wx;           matrix[8] = T(1) - (xx + yy);
}
```

---

## Task 11: matrix_to_quaternion

Convert 3x3 rotation matrix to quaternion using Shepperd's method for numerical stability.

**Kernel** (`matrix_to_quaternion.h`):
```cpp
template <typename T>
void matrix_to_quaternion_scalar(const T* m, T* q) {
  // Shepperd's method - choose largest diagonal element for numerical stability
  const T m00 = m[0], m01 = m[1], m02 = m[2];
  const T m10 = m[3], m11 = m[4], m12 = m[5];
  const T m20 = m[6], m21 = m[7], m22 = m[8];

  const T trace = m00 + m11 + m22;

  if (trace > T(0)) {
    const T s = std::sqrt(trace + T(1)) * T(2);
    q[0] = T(0.25) * s;
    q[1] = (m21 - m12) / s;
    q[2] = (m02 - m20) / s;
    q[3] = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const T s = std::sqrt(T(1) + m00 - m11 - m22) * T(2);
    q[0] = (m21 - m12) / s;
    q[1] = T(0.25) * s;
    q[2] = (m01 + m10) / s;
    q[3] = (m02 + m20) / s;
  } else if (m11 > m22) {
    const T s = std::sqrt(T(1) + m11 - m00 - m22) * T(2);
    q[0] = (m02 - m20) / s;
    q[1] = (m01 + m10) / s;
    q[2] = T(0.25) * s;
    q[3] = (m12 + m21) / s;
  } else {
    const T s = std::sqrt(T(1) + m22 - m00 - m11) * T(2);
    q[0] = (m10 - m01) / s;
    q[1] = (m02 + m20) / s;
    q[2] = (m12 + m21) / s;
    q[3] = T(0.25) * s;
  }
}
```

---

## Task 12: quaternion_slerp

Spherical linear interpolation between quaternions.

**Kernel** (`quaternion_slerp.h`):
```cpp
template <typename T>
void quaternion_slerp_scalar(const T* q1, const T* q2, T t, T* output) {
  // Compute cosine of angle between quaternions
  T dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3];

  // If dot is negative, negate one quaternion to take shorter path
  T q2_adj[4] = {q2[0], q2[1], q2[2], q2[3]};
  if (dot < T(0)) {
    dot = -dot;
    q2_adj[0] = -q2[0];
    q2_adj[1] = -q2[1];
    q2_adj[2] = -q2[2];
    q2_adj[3] = -q2[3];
  }

  // If quaternions are very close, use linear interpolation
  if (dot > T(0.9995)) {
    output[0] = q1[0] + t * (q2_adj[0] - q1[0]);
    output[1] = q1[1] + t * (q2_adj[1] - q1[1]);
    output[2] = q1[2] + t * (q2_adj[2] - q1[2]);
    output[3] = q1[3] + t * (q2_adj[3] - q1[3]);
    // Normalize
    T norm = std::sqrt(output[0]*output[0] + output[1]*output[1] +
                       output[2]*output[2] + output[3]*output[3]);
    output[0] /= norm;
    output[1] /= norm;
    output[2] /= norm;
    output[3] /= norm;
  } else {
    const T theta = std::acos(dot);
    const T sin_theta = std::sin(theta);
    const T s1 = std::sin((T(1) - t) * theta) / sin_theta;
    const T s2 = std::sin(t * theta) / sin_theta;

    output[0] = s1 * q1[0] + s2 * q2_adj[0];
    output[1] = s1 * q1[1] + s2 * q2_adj[1];
    output[2] = s1 * q1[2] + s2 * q2_adj[2];
    output[3] = s1 * q1[3] + s2 * q2_adj[3];
  }
}
```

---

## Task 13: RotationMatrix Tensorclass

**Files:**
- Create: `src/torchscience/geometry/transform/_rotation_matrix.py`
- Modify: `src/torchscience/geometry/transform/__init__.py`
- Test: `tests/torchscience/geometry/transform/test__rotation_matrix.py`

Similar to Quaternion tensorclass:

```python
@tensorclass
class RotationMatrix:
    """3x3 rotation matrix (SO(3) element).

    Attributes
    ----------
    matrix : Tensor
        Rotation matrix, shape (..., 3, 3). Should be orthogonal with det = +1.
    """
    matrix: Tensor
```

---

## Task 14: Final Integration Tests

Add scipy comparison tests:

```python
def test_scipy_comparison():
    """Compare with scipy.spatial.transform.Rotation."""
    from scipy.spatial.transform import Rotation as R

    # Random quaternion
    q_scipy = R.random()
    q_wxyz = torch.tensor(q_scipy.as_quat()[[3, 0, 1, 2]], dtype=torch.float64)  # scipy uses xyzw
    q = quaternion(q_wxyz)

    # Compare matrix conversion
    mat_scipy = torch.tensor(q_scipy.as_matrix(), dtype=torch.float64)
    mat_torch = quaternion_to_matrix(q).matrix
    assert torch.allclose(mat_torch, mat_scipy, atol=1e-6)

    # Compare point rotation
    point = torch.randn(3, dtype=torch.float64)
    rotated_scipy = torch.tensor(q_scipy.apply(point.numpy()), dtype=torch.float64)
    rotated_torch = quaternion_apply(q, point)
    assert torch.allclose(rotated_torch, rotated_scipy, atol=1e-6)
```

---

## Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | Quaternion tensorclass | 3 files |
| 2 | quaternion_multiply schema+kernel | 3 files |
| 3 | quaternion_multiply CPU | 2 files |
| 4 | quaternion_multiply Meta | 2 files |
| 5 | quaternion_multiply Autograd | 2 files |
| 6 | quaternion_multiply Python+tests | 3 files |
| 7 | quaternion_inverse (full stack) | 8 files |
| 8 | quaternion_normalize (full stack) | 8 files |
| 9 | quaternion_apply (full stack) | 8 files |
| 10 | quaternion_to_matrix (full stack) | 8 files |
| 11 | matrix_to_quaternion (full stack) | 8 files |
| 12 | quaternion_slerp (full stack) | 8 files |
| 13 | RotationMatrix tensorclass | 3 files |
| 14 | Integration tests | 1 file |

Total: ~14 tasks, ~70 files created/modified
