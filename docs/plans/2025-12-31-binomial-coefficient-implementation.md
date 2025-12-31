# Binomial Coefficient Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `torchscience.combinatorics.binomial_coefficient` - a batched, differentiable generalized binomial coefficient using the gamma function representation.

**Architecture:** The binomial coefficient C(n,k) = Gamma(n+1) / (Gamma(k+1) * Gamma(n-k+1)) is implemented as a binary pointwise operator following the existing special_functions pattern. Computation uses log-gamma for numerical stability. Full autograd support via digamma/trigamma for first/second-order gradients.

**Tech Stack:** C++20, PyTorch ATen, existing kernel infrastructure (log_gamma, digamma, trigamma)

---

## Task 1: Create Kernel Forward Implementation

**Files:**
- Create: `src/torchscience/csrc/kernel/combinatorics/binomial_coefficient.h`

**Step 1: Create the kernel header with forward implementation**

```cpp
#pragma once

#include <cmath>
#include "../special_functions/log_gamma.h"

namespace torchscience::kernel::combinatorics {

template <typename T>
T binomial_coefficient(T n, T k) {
  // C(n, k) = 0 for k < 0 or (n >= 0 and k > n)
  if (k < T(0)) {
    return T(0);
  }

  if (n >= T(0) && k > n) {
    return T(0);
  }

  // C(n, 0) = 1 for all n
  if (k == T(0)) {
    return T(1);
  }

  // Use log-gamma for numerical stability:
  // C(n, k) = exp(lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1))
  T log_result = special_functions::log_gamma(n + T(1))
               - special_functions::log_gamma(k + T(1))
               - special_functions::log_gamma(n - k + T(1));

  return std::exp(log_result);
}

} // namespace torchscience::kernel::combinatorics
```

**Step 2: Verify file compiles (syntax check)**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience && ls src/torchscience/csrc/kernel/combinatorics/binomial_coefficient.h`
Expected: File exists

**Step 3: Commit**

```bash
git add src/torchscience/csrc/kernel/combinatorics/binomial_coefficient.h
git commit -m "feat(kernel): add binomial_coefficient forward kernel"
```

---

## Task 2: Create Kernel Backward Implementation

**Files:**
- Create: `src/torchscience/csrc/kernel/combinatorics/binomial_coefficient_backward.h`

**Step 1: Create the backward kernel**

The gradient of C(n,k) with respect to n and k:
- d/dn C(n,k) = C(n,k) * (digamma(n+1) - digamma(n-k+1))
- d/dk C(n,k) = C(n,k) * (-digamma(k+1) + digamma(n-k+1))

```cpp
#pragma once

#include <tuple>
#include <cmath>
#include "binomial_coefficient.h"
#include "../special_functions/digamma.h"

namespace torchscience::kernel::combinatorics {

template <typename T>
std::tuple<T, T> binomial_coefficient_backward(T grad_output, T n, T k) {
  // For edge cases where C(n,k) = 0, gradients are 0
  if (k < T(0) || (n >= T(0) && k > n)) {
    return {T(0), T(0)};
  }

  // For k = 0, C(n,0) = 1 constant, so gradients are 0
  if (k == T(0)) {
    return {T(0), T(0)};
  }

  T c_nk = binomial_coefficient(n, k);

  T psi_n_plus_1 = special_functions::digamma(n + T(1));
  T psi_k_plus_1 = special_functions::digamma(k + T(1));
  T psi_n_minus_k_plus_1 = special_functions::digamma(n - k + T(1));

  // d/dn C(n,k) = C(n,k) * (psi(n+1) - psi(n-k+1))
  T grad_n = grad_output * c_nk * (psi_n_plus_1 - psi_n_minus_k_plus_1);

  // d/dk C(n,k) = C(n,k) * (-psi(k+1) + psi(n-k+1))
  T grad_k = grad_output * c_nk * (-psi_k_plus_1 + psi_n_minus_k_plus_1);

  return {grad_n, grad_k};
}

} // namespace torchscience::kernel::combinatorics
```

**Step 2: Verify file exists**

Run: `ls src/torchscience/csrc/kernel/combinatorics/binomial_coefficient_backward.h`
Expected: File exists

**Step 3: Commit**

```bash
git add src/torchscience/csrc/kernel/combinatorics/binomial_coefficient_backward.h
git commit -m "feat(kernel): add binomial_coefficient backward kernel"
```

---

## Task 3: Create Kernel Backward-Backward Implementation

**Files:**
- Create: `src/torchscience/csrc/kernel/combinatorics/binomial_coefficient_backward_backward.h`

**Step 1: Create the second-order backward kernel**

```cpp
#pragma once

#include <tuple>
#include <cmath>
#include "binomial_coefficient.h"
#include "../special_functions/digamma.h"
#include "../special_functions/trigamma.h"

namespace torchscience::kernel::combinatorics {

template <typename T>
std::tuple<T, T, T> binomial_coefficient_backward_backward(
  T gg_n,
  T gg_k,
  T grad_output,
  T n,
  T k
) {
  // For edge cases where C(n,k) = 0 or constant, second derivatives are 0
  if (k < T(0) || (n >= T(0) && k > n) || k == T(0)) {
    return {T(0), T(0), T(0)};
  }

  T c_nk = binomial_coefficient(n, k);

  T psi_n1 = special_functions::digamma(n + T(1));
  T psi_k1 = special_functions::digamma(k + T(1));
  T psi_nmk1 = special_functions::digamma(n - k + T(1));

  T psi1_n1 = special_functions::trigamma(n + T(1));
  T psi1_k1 = special_functions::trigamma(k + T(1));
  T psi1_nmk1 = special_functions::trigamma(n - k + T(1));

  T dn = psi_n1 - psi_nmk1;
  T dk = -psi_k1 + psi_nmk1;

  // Second derivatives:
  // d2/dn2 C = C * (dn^2 + psi1(n+1) - psi1(n-k+1))
  // d2/dk2 C = C * (dk^2 + psi1(k+1) + psi1(n-k+1))
  // d2/dndk C = C * (dn*dk - psi1(n-k+1))

  T d2_nn = c_nk * (dn * dn + psi1_n1 - psi1_nmk1);
  T d2_kk = c_nk * (dk * dk + psi1_k1 + psi1_nmk1);
  T d2_nk = c_nk * (dn * dk - psi1_nmk1);

  // grad_grad_output = gg_n * d(grad_n)/d(grad_output) + gg_k * d(grad_k)/d(grad_output)
  //                  = gg_n * C * dn + gg_k * C * dk
  T grad_grad_output = gg_n * c_nk * dn + gg_k * c_nk * dk;

  // grad_n from backward_backward:
  // d/dn of (grad_output * C * dn) w.r.t. n, summed with cross terms
  T grad_n = grad_output * (gg_n * d2_nn + gg_k * d2_nk);

  // grad_k from backward_backward:
  T grad_k = grad_output * (gg_n * d2_nk + gg_k * d2_kk);

  return {grad_grad_output, grad_n, grad_k};
}

} // namespace torchscience::kernel::combinatorics
```

**Step 2: Verify file exists**

Run: `ls src/torchscience/csrc/kernel/combinatorics/binomial_coefficient_backward_backward.h`
Expected: File exists

**Step 3: Commit**

```bash
git add src/torchscience/csrc/kernel/combinatorics/binomial_coefficient_backward_backward.h
git commit -m "feat(kernel): add binomial_coefficient backward_backward kernel"
```

---

## Task 4: Create CPU Backend

**Files:**
- Create: `src/torchscience/csrc/cpu/combinatorics.h`

**Step 1: Create CPU backend using existing macros**

```cpp
#pragma once

#include "macros.h"

#include "../kernel/combinatorics/binomial_coefficient.h"
#include "../kernel/combinatorics/binomial_coefficient_backward.h"
#include "../kernel/combinatorics/binomial_coefficient_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(binomial_coefficient, n, k)
```

**Step 2: Verify file exists**

Run: `ls src/torchscience/csrc/cpu/combinatorics.h`
Expected: File exists

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cpu/combinatorics.h
git commit -m "feat(cpu): add binomial_coefficient CPU backend"
```

---

## Task 5: Create Meta Backend

**Files:**
- Create: `src/torchscience/csrc/meta/combinatorics.h`

**Step 1: Create Meta backend for shape inference**

```cpp
#pragma once

#include "macros.h"

TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(binomial_coefficient, n, k)
```

**Step 2: Verify file exists**

Run: `ls src/torchscience/csrc/meta/combinatorics.h`
Expected: File exists

**Step 3: Commit**

```bash
git add src/torchscience/csrc/meta/combinatorics.h
git commit -m "feat(meta): add binomial_coefficient meta backend"
```

---

## Task 6: Create Autograd Backend

**Files:**
- Create: `src/torchscience/csrc/autograd/combinatorics.h`

**Step 1: Create Autograd backend**

```cpp
#pragma once

#include "macros.h"

TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(binomial_coefficient, BinomialCoefficient, n, k)
```

**Step 2: Verify file exists**

Run: `ls src/torchscience/csrc/autograd/combinatorics.h`
Expected: File exists

**Step 3: Commit**

```bash
git add src/torchscience/csrc/autograd/combinatorics.h
git commit -m "feat(autograd): add binomial_coefficient autograd backend"
```

---

## Task 7: Create Autocast Backend

**Files:**
- Create: `src/torchscience/csrc/autocast/combinatorics.h`

**Step 1: Check existing autocast pattern**

Run: `head -30 src/torchscience/csrc/autocast/special_functions.h`

**Step 2: Create Autocast backend**

```cpp
#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::combinatorics {

inline at::Tensor binomial_coefficient(
  const at::Tensor &n_input,
  const at::Tensor &k_input
) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

  auto target_dtype = at::promote_types(n_input.scalar_type(), k_input.scalar_type());

  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::binomial_coefficient", "")
    .typed<at::Tensor(const at::Tensor &, const at::Tensor &)>()
    .call(
      at::autocast::cached_cast(target_dtype, n_input, c10::DeviceType::CUDA),
      at::autocast::cached_cast(target_dtype, k_input, c10::DeviceType::CUDA)
    );
}

} // namespace torchscience::autocast::combinatorics

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
  module.impl(
    "binomial_coefficient",
    torchscience::autocast::combinatorics::binomial_coefficient
  );
}
```

**Step 3: Verify file exists**

Run: `ls src/torchscience/csrc/autocast/combinatorics.h`
Expected: File exists

**Step 4: Commit**

```bash
git add src/torchscience/csrc/autocast/combinatorics.h
git commit -m "feat(autocast): add binomial_coefficient autocast backend"
```

---

## Task 8: Register Schema in torchscience.cpp

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp`

**Step 1: Add includes at top of file (after existing includes)**

Find the include section and add:

```cpp
#include "cpu/combinatorics.h"
#include "meta/combinatorics.h"
#include "autograd/combinatorics.h"
#include "autocast/combinatorics.h"
```

**Step 2: Add schema definitions in TORCH_LIBRARY block**

Find the `TORCH_LIBRARY(torchscience, module)` block and add before the closing brace:

```cpp
  // combinatorics
  module.def("binomial_coefficient(Tensor n, Tensor k) -> Tensor");
  module.def("binomial_coefficient_backward(Tensor grad_output, Tensor n, Tensor k) -> (Tensor, Tensor)");
  module.def("binomial_coefficient_backward_backward(Tensor gg_n, Tensor gg_k, Tensor grad_output, Tensor n, Tensor k) -> (Tensor, Tensor, Tensor)");
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat: register binomial_coefficient operator schema"
```

---

## Task 9: Create Python API

**Files:**
- Create: `src/torchscience/combinatorics/_binomial_coefficient.py`

**Step 1: Create Python wrapper**

```python
import torch
from torch import Tensor


def binomial_coefficient(n: Tensor, k: Tensor) -> Tensor:
    r"""
    Binomial coefficient.

    Computes the generalized binomial coefficient C(n, k) = n! / (k! * (n-k)!)
    using the gamma function representation for numerical stability and to
    support non-integer arguments.

    Mathematical Definition
    -----------------------
    The binomial coefficient is defined as:

    .. math::

       \binom{n}{k} = \frac{\Gamma(n+1)}{\Gamma(k+1) \Gamma(n-k+1)}

    For non-negative integers, this equals the number of ways to choose
    k items from n items without replacement.

    Special Values
    --------------
    - C(n, 0) = 1 for all n
    - C(n, n) = 1 for non-negative integer n
    - C(n, k) = 0 for k < 0 or (n >= 0 and k > n)
    - C(n, k) = C(n, n-k) (symmetry for non-negative integer n)

    Generalized Binomial Coefficients
    ---------------------------------
    For non-integer n, this computes the generalized binomial coefficient:

    .. math::

       \binom{n}{k} = \frac{n(n-1)(n-2)\cdots(n-k+1)}{k!}

    This extends the binomial coefficient to negative and fractional n,
    which appears in the binomial series expansion of (1+x)^n.

    Autograd Support
    ----------------
    Gradients are fully supported when n.requires_grad or k.requires_grad
    is True. The gradients are computed using the digamma function:

    .. math::

       \frac{\partial}{\partial n} \binom{n}{k} = \binom{n}{k}
           \left( \psi(n+1) - \psi(n-k+1) \right)

    .. math::

       \frac{\partial}{\partial k} \binom{n}{k} = \binom{n}{k}
           \left( -\psi(k+1) + \psi(n-k+1) \right)

    Second-order derivatives are also supported using the trigamma function.

    Parameters
    ----------
    n : Tensor
        Number of items to choose from. Can be any real number for
        generalized binomial coefficients.
    k : Tensor
        Number of items to choose. Must be broadcastable with n.

    Returns
    -------
    Tensor
        The binomial coefficient C(n, k) for each element pair.
        Output shape is the broadcast shape of n and k.

    Examples
    --------
    Integer binomial coefficients (Pascal's triangle):

    >>> n = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    >>> k = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    >>> binomial_coefficient(n, k)
    tensor([ 1.,  5., 10., 10.,  5.,  1.])

    Generalized binomial coefficient with negative n:

    >>> n = torch.tensor([-0.5])
    >>> k = torch.tensor([2.0])
    >>> binomial_coefficient(n, k)
    tensor([0.375])  # (-0.5)(-1.5) / 2! = 0.75 / 2 = 0.375

    Autograd example:

    >>> n = torch.tensor([5.0], requires_grad=True)
    >>> k = torch.tensor([2.0])
    >>> result = binomial_coefficient(n, k)
    >>> result.backward()
    >>> n.grad
    tensor([...])  # gradient w.r.t. n

    See Also
    --------
    torchscience.special_functions.gamma : Gamma function
    torchscience.special_functions.log_gamma : Log-gamma function
    scipy.special.comb : SciPy's combination function
    """
    return torch.ops.torchscience.binomial_coefficient(n, k)
```

**Step 2: Verify file exists**

Run: `ls src/torchscience/combinatorics/_binomial_coefficient.py`
Expected: File exists

**Step 3: Commit**

```bash
git add src/torchscience/combinatorics/_binomial_coefficient.py
git commit -m "feat(combinatorics): add binomial_coefficient Python API"
```

---

## Task 10: Update combinatorics __init__.py

**Files:**
- Modify: `src/torchscience/combinatorics/__init__.py`

**Step 1: Add export**

```python
from torchscience.combinatorics._binomial_coefficient import binomial_coefficient

__all__ = [
    "binomial_coefficient",
]
```

**Step 2: Commit**

```bash
git add src/torchscience/combinatorics/__init__.py
git commit -m "feat(combinatorics): export binomial_coefficient"
```

---

## Task 11: Write Tests - Forward Correctness

**Files:**
- Create: `tests/torchscience/combinatorics/__init__.py`
- Create: `tests/torchscience/combinatorics/test__binomial_coefficient.py`

**Step 1: Create test directory init file**

```python
```

**Step 2: Create test file with forward tests**

```python
import math

import pytest
import torch
import torch.testing

import torchscience.combinatorics


class TestBinomialCoefficient:
    """Tests for the binomial coefficient function."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_pascals_triangle_row_5(self):
        """Test C(5, k) for k=0..5 matches Pascal's triangle."""
        n = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float64)
        k = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.tensor([1.0, 5.0, 10.0, 10.0, 5.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_against_scipy_formula(self):
        """Test against gamma function formula."""
        n = torch.tensor([10.0, 7.0, 20.0], dtype=torch.float64)
        k = torch.tensor([3.0, 4.0, 10.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.exp(
            torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_c_n_0_equals_1(self):
        """Test C(n, 0) = 1 for various n."""
        n = torch.tensor([0.0, 1.0, 5.0, 10.0, 100.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.ones(5, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_c_n_n_equals_1(self):
        """Test C(n, n) = 1 for non-negative integer n."""
        n = torch.tensor([0.0, 1.0, 5.0, 10.0], dtype=torch.float64)
        k = n.clone()
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.ones(4, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_c_n_1_equals_n(self):
        """Test C(n, 1) = n."""
        n = torch.tensor([1.0, 2.0, 5.0, 10.0, 100.0], dtype=torch.float64)
        k = torch.ones(5, dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        torch.testing.assert_close(result, n, rtol=1e-10, atol=1e-10)

    def test_symmetry(self):
        """Test C(n, k) = C(n, n-k)."""
        n = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        k = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        result_k = torchscience.combinatorics.binomial_coefficient(n, k)
        result_n_minus_k = torchscience.combinatorics.binomial_coefficient(n, n - k)
        torch.testing.assert_close(result_k, result_n_minus_k, rtol=1e-10, atol=1e-10)

    def test_k_negative_returns_zero(self):
        """Test C(n, k) = 0 for k < 0."""
        n = torch.tensor([5.0, 10.0], dtype=torch.float64)
        k = torch.tensor([-1.0, -2.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.zeros(2, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_k_greater_than_n_returns_zero(self):
        """Test C(n, k) = 0 for k > n when n >= 0."""
        n = torch.tensor([5.0, 3.0], dtype=torch.float64)
        k = torch.tensor([6.0, 10.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.zeros(2, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_generalized_negative_n(self):
        """Test generalized binomial coefficient with negative n."""
        # C(-0.5, 2) = (-0.5)(-1.5) / 2! = 0.75 / 2 = 0.375
        n = torch.tensor([-0.5], dtype=torch.float64)
        k = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.tensor([0.375], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_generalized_fractional_n(self):
        """Test generalized binomial coefficient with fractional n."""
        # C(0.5, 2) = (0.5)(-0.5) / 2! = -0.25 / 2 = -0.125
        n = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        expected = torch.tensor([-0.125], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test forward pass for float dtypes."""
        n = torch.tensor([5.0, 10.0], dtype=dtype)
        k = torch.tensor([2.0, 3.0], dtype=dtype)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        assert result.dtype == dtype
        expected = torch.tensor([10.0, 120.0], dtype=dtype)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test broadcasting behavior."""
        n = torch.tensor([[5.0], [10.0]], dtype=torch.float64)  # (2, 1)
        k = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)  # (3,)
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        assert result.shape == (2, 3)
        expected = torch.tensor(
            [[1.0, 5.0, 10.0], [1.0, 10.0, 45.0]], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        n = torch.tensor([5.0, 10.0, 7.0], dtype=torch.float64, requires_grad=True)
        k = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)

        def func(n, k):
            return torchscience.combinatorics.binomial_coefficient(n, k)

        assert torch.autograd.gradcheck(func, (n, k), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        n = torch.tensor([5.0, 10.0], dtype=torch.float64, requires_grad=True)
        k = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)

        def func(n, k):
            return torchscience.combinatorics.binomial_coefficient(n, k)

        assert torch.autograd.gradgradcheck(
            func, (n, k), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test with meta tensors (shape inference only)."""
        n = torch.empty(3, 4, device="meta")
        k = torch.empty(3, 4, device="meta")
        result = torchscience.combinatorics.binomial_coefficient(n, k)
        assert result.device.type == "meta"
        assert result.shape == (3, 4)
```

**Step 3: Run tests to verify they fail (module not yet built)**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience && uv run pytest tests/torchscience/combinatorics/test__binomial_coefficient.py -v --tb=short 2>&1 | head -50`
Expected: Tests fail (import error or operator not found)

**Step 4: Commit**

```bash
git add tests/torchscience/combinatorics/__init__.py tests/torchscience/combinatorics/test__binomial_coefficient.py
git commit -m "test(combinatorics): add binomial_coefficient tests"
```

---

## Task 12: Build and Run Tests

**Files:**
- None (verification only)

**Step 1: Rebuild the extension**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience && uv run pip install -e . --no-build-isolation 2>&1 | tail -20`
Expected: Build succeeds

**Step 2: Run binomial_coefficient tests**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience && uv run pytest tests/torchscience/combinatorics/test__binomial_coefficient.py -v`
Expected: All tests pass

**Step 3: Run full test suite to check for regressions**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience && uv run pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: No new failures

---

## Task 13: Final Commit

**Step 1: Verify all changes are committed**

Run: `git status`
Expected: Clean working tree

**Step 2: If any uncommitted changes, commit them**

```bash
git add -A
git commit -m "feat(combinatorics): complete binomial_coefficient implementation"
```
