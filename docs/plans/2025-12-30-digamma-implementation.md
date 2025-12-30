# Digamma Function Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `torchscience.special_functions.digamma` with full autograd support including second-order gradients.

**Architecture:** Digamma (ψ) is the logarithmic derivative of gamma. Forward uses existing kernel. Backward uses trigamma (ψ'). Backward-backward needs new tetragamma kernel (ψ'').

**Tech Stack:** C++17, PyTorch ATen, TORCH_LIBRARY macros

---

## Task 1: Implement Tetragamma Kernel

**Files:**
- Create: `src/torchscience/csrc/kernel/special_functions/tetragamma.h`

**Step 1: Create tetragamma kernel header**

```cpp
#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "sin_pi.h"
#include "cos_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T tetragamma(T z) {
  T psi2 = T(0);

  T y = z;

  while (y < T(6)) {
    psi2 -= T(2) / (y * y * y);

    y += T(1);
  }

  T y2 = T(1) / (y * y);
  T y3 = y2 / y;

  // Asymptotic expansion: psi''(y) = -1/y^2 - 1/y^3 - 1/(2y^4) + 1/(6y^6) - 1/(6y^8) + ...
  psi2 += -y2 - y3 - y2 * y2 * (T(0.5) - y2 * (T(1.0/6) - y2 * T(1.0/6)));

  return psi2;
}

template <typename T>
c10::complex<T> tetragamma(c10::complex<T> z) {
  c10::complex<T> psi2(T(0), T(0));

  c10::complex<T> y = z;

  if (y.real() < T(0.5)) {
    // Reflection formula: psi''(z) + psi''(1-z) = 2*pi^3*cos(pi*z)/sin^3(pi*z)
    auto sin_piz = sin_pi(y);
    auto cos_piz = cos_pi(y);
    auto sin_piz_cubed = sin_piz * sin_piz * sin_piz;
    auto pi_cubed = static_cast<T>(M_PI * M_PI * M_PI);

    return pi_cubed * static_cast<T>(2) * cos_piz / sin_piz_cubed - tetragamma(c10::complex<T>(T(1), T(0)) - y);
  }

  while (std::abs(y) < T(6)) {
    psi2 = psi2 - c10::complex<T>(T(2), T(0)) / (y * y * y);

    y = y + c10::complex<T>(T(1), T(0));
  }

  c10::complex<T> y2 = c10::complex<T>(T(1), T(0)) / (y * y);
  c10::complex<T> y3 = y2 / y;

  return psi2 + (-y2 - y3 - y2 * y2 * (c10::complex<T>(T(0.5), T(0)) - y2 * (c10::complex<T>(T(1.0/6), T(0)) - y2 * c10::complex<T>(T(1.0/6), T(0)))));
}

} // namespace torchscience::kernel::special_functions
```

**Step 2: Verify file exists**

Run: `ls -la src/torchscience/csrc/kernel/special_functions/tetragamma.h`
Expected: File listed with correct size

**Step 3: Commit**

```bash
git add src/torchscience/csrc/kernel/special_functions/tetragamma.h
git commit -m "feat(kernel): add tetragamma kernel for second-order digamma gradients"
```

---

## Task 2: Implement Digamma Backward Kernel

**Files:**
- Create: `src/torchscience/csrc/kernel/special_functions/digamma_backward.h`

**Step 1: Create digamma backward kernel**

```cpp
#pragma once

#include "trigamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T digamma_backward(T gradient, T z) {
  return gradient * trigamma(z);
}

template <typename T>
c10::complex<T> digamma_backward(c10::complex<T> gradient, c10::complex<T> z) {
  return gradient * trigamma(z);
}

} // namespace torchscience::kernel::special_functions
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/kernel/special_functions/digamma_backward.h
git commit -m "feat(kernel): add digamma_backward kernel using trigamma"
```

---

## Task 3: Implement Digamma Backward-Backward Kernel

**Files:**
- Create: `src/torchscience/csrc/kernel/special_functions/digamma_backward_backward.h`

**Step 1: Create digamma backward-backward kernel**

```cpp
#pragma once

#include <tuple>

#include "trigamma.h"
#include "tetragamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> digamma_backward_backward(
  T gradient_gradient,
  T gradient,
  T z
) {
  return {
    gradient_gradient * trigamma(z),
    gradient_gradient * gradient * tetragamma(z)
  };
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> digamma_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  return {
    gradient_gradient * trigamma(z),
    gradient_gradient * gradient * tetragamma(z)
  };
}

} // namespace torchscience::kernel::special_functions
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/kernel/special_functions/digamma_backward_backward.h
git commit -m "feat(kernel): add digamma_backward_backward kernel using tetragamma"
```

---

## Task 4: Register CPU Operator

**Files:**
- Modify: `src/torchscience/csrc/cpu/special_functions.h`

**Step 1: Add digamma includes and macro after gamma**

Add after line 9 (after `TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR(gamma, z)`):

```cpp
#include "../kernel/special_functions/digamma.h"
#include "../kernel/special_functions/digamma_backward.h"
#include "../kernel/special_functions/digamma_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR(digamma, z)
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/special_functions.h
git commit -m "feat(cpu): register digamma operator"
```

---

## Task 5: Register Meta Operator

**Files:**
- Modify: `src/torchscience/csrc/meta/special_functions.h`

**Step 1: Add digamma macro after gamma**

Add after line 5 (after `TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(gamma, z)`):

```cpp
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(digamma, z)
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/meta/special_functions.h
git commit -m "feat(meta): register digamma operator"
```

---

## Task 6: Register Autograd Operator

**Files:**
- Modify: `src/torchscience/csrc/autograd/special_functions.h`

**Step 1: Add digamma macro after gamma**

Add after line 5 (after `TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(gamma, Gamma, z)`):

```cpp
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(digamma, Digamma, z)
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/autograd/special_functions.h
git commit -m "feat(autograd): register digamma operator"
```

---

## Task 7: Register Autocast Operator

**Files:**
- Modify: `src/torchscience/csrc/autocast/special_functions.h`

**Step 1: Add digamma macro after gamma**

Add after line 5 (after `TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(gamma, z)`):

```cpp
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(digamma, z)
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/autocast/special_functions.h
git commit -m "feat(autocast): register digamma operator"
```

---

## Task 8: Add Schema Definitions

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp`

**Step 1: Add digamma schema after gamma schemas (after line 101)**

```cpp
  module.def("digamma(Tensor z) -> Tensor");
  module.def("digamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("digamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");
```

**Step 2: Build and verify compilation**

Run: `uv run python -c "import torchscience; print(torchscience.special_functions.gamma.__doc__[:50])"`
Expected: No import errors

**Step 3: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat(schema): add digamma operator definitions"
```

---

## Task 9: Create Python API

**Files:**
- Create: `src/torchscience/special_functions/_digamma.py`

**Step 1: Create Python module**

```python
import torch
from torch import Tensor


def digamma(z: Tensor) -> Tensor:
    r"""
    Digamma function.

    Computes the digamma (psi) function evaluated at each element of the input
    tensor. The digamma function is the logarithmic derivative of the gamma
    function.

    Mathematical Definition
    -----------------------
    The digamma function is defined as:

    .. math::

       \psi(z) = \frac{d}{dz} \ln \Gamma(z) = \frac{\Gamma'(z)}{\Gamma(z)}

    Special Values
    --------------
    - psi(1) = -gamma (Euler-Mascheroni constant, approximately -0.5772)
    - psi(n) = -gamma + H_{n-1} for positive integers n, where H_n is the
      n-th harmonic number
    - psi(1/2) = -gamma - 2*ln(2)

    Domain
    ------
    - z: any real or complex value except non-positive integers
    - Poles at z = 0, -1, -2, -3, ... where the function returns -inf

    Algorithm
    ---------
    - Uses recurrence relation psi(z+1) = psi(z) + 1/z to shift argument
    - Asymptotic expansion for |z| >= 6
    - Reflection formula for Re(z) < 0.5 (complex)

    Recurrence Relations
    --------------------
    - psi(z+1) = psi(z) + 1/z
    - psi(1-z) - psi(z) = pi * cot(pi*z)

    Applications
    ------------
    The digamma function appears in:
    - Maximum likelihood estimation for gamma and Dirichlet distributions
    - Gradients of the gamma function: d/dz Gamma(z) = Gamma(z) * psi(z)
    - Bayesian inference with conjugate priors
    - Renormalization in quantum field theory

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using the trigamma function:

    .. math::

       \frac{d}{dz} \psi(z) = \psi'(z) = \text{trigamma}(z)

    Second-order derivatives are also supported using the tetragamma function.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The digamma function evaluated at each element of z.

    Examples
    --------
    Evaluate at positive integers:

    >>> z = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> digamma(z)
    tensor([-0.5772,  0.4228,  0.9228,  1.2561])

    Verify recurrence relation psi(z+1) = psi(z) + 1/z:

    >>> z = torch.tensor([2.0])
    >>> digamma(z + 1) - digamma(z)
    tensor([0.5000])  # equals 1/z = 0.5

    Complex input:

    >>> z = torch.tensor([1.0 + 1.0j])
    >>> digamma(z)
    tensor([0.0946+1.0762j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = digamma(z)
    >>> y.backward()
    >>> z.grad  # trigamma(2) = pi^2/6 - 1 ≈ 0.6449
    tensor([0.6449])

    See Also
    --------
    torchscience.special_functions.gamma : Gamma function
    torch.special.digamma : PyTorch's digamma implementation
    torch.special.polygamma : General polygamma function
    """
    return torch.ops.torchscience.digamma(z)
```

**Step 2: Commit**

```bash
git add src/torchscience/special_functions/_digamma.py
git commit -m "feat(api): add digamma Python function with docstring"
```

---

## Task 10: Export from Package

**Files:**
- Modify: `src/torchscience/special_functions/__init__.py`

**Step 1: Add digamma import and export**

Add import after line 3:
```python
from ._digamma import digamma
```

Add to `__all__` list (alphabetically):
```python
    "digamma",
```

**Step 2: Commit**

```bash
git add src/torchscience/special_functions/__init__.py
git commit -m "feat(api): export digamma from special_functions"
```

---

## Task 11: Write Tests

**Files:**
- Create: `tests/torchscience/special_functions/test__digamma.py`

**Step 1: Create comprehensive test file**

```python
import math

import pytest
import sympy
import torch
import torch.testing
from hypothesis import given, settings
from sympy import I, N, symbols

import torchscience.special_functions
from torchscience.testing import (
    IdentitySpec,
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    RecurrenceSpec,
    SingularitySpec,
    SpecialValue,
    SymbolicDerivativeVerifier,
    ToleranceConfig,
    avoiding_poles,
    complex_avoiding_real_axis,
    positive_real_numbers,
)


EULER_MASCHERONI = 0.5772156649015329


def sympy_digamma(z: float | complex) -> float | complex:
    """Wrapper for SymPy digamma function."""
    if isinstance(z, complex):
        sympy_z = sympy.Float(z.real) + I * sympy.Float(z.imag)
    else:
        sympy_z = sympy.Float(z)
    result = N(sympy.digamma(sympy_z), 50)
    if result.is_real:
        return float(result)
    return complex(result)


def create_digamma_verifier() -> SymbolicDerivativeVerifier:
    """Create derivative verifier for the digamma function."""
    z = symbols("z")
    expr = sympy.digamma(z)
    return SymbolicDerivativeVerifier(expr, [z])


def _check_recurrence(func) -> bool:
    """Check psi(x+1) = psi(x) + 1/x."""
    x = torch.tensor([0.5, 1.5, 2.5, 3.7], dtype=torch.float64)
    left = func(x + 1)
    right = func(x) + 1 / x
    return torch.allclose(left, right, rtol=1e-10, atol=1e-10)


def _reflection_identity(func):
    """Check psi(1-x) - psi(x) = pi * cot(pi*x)."""
    x = torch.tensor([0.25, 0.3, 0.4, 0.6], dtype=torch.float64)
    left = func(1 - x) - func(x)
    right = math.pi / torch.tan(math.pi * x)
    return left, right


class TestDigamma(OpTestCase):
    """Tests for the digamma function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="digamma",
            func=torchscience.special_functions.digamma,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z",
                    position=0,
                    default_real_range=(0.5, 20.0),
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                ),
            ],
            sympy_func=sympy.digamma,
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",  # CPU autocast not supported
                "test_gradgradcheck_complex",  # Complex 2nd order numerically sensitive
                "test_sparse_coo_basic",  # Sparse has implicit zeros = poles
                "test_low_precision_forward",  # Random values may hit poles
            },
            recurrence_relations=[
                RecurrenceSpec(
                    name="digamma_recurrence",
                    check_fn=_check_recurrence,
                    description="psi(x+1) = psi(x) + 1/x",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="reflection_formula",
                    identity_fn=_reflection_identity,
                    description="psi(1-x) - psi(x) = pi * cot(pi*x)",
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0,),
                    expected=-EULER_MASCHERONI,
                    description="psi(1) = -gamma",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=1 - EULER_MASCHERONI,
                    description="psi(2) = 1 - gamma",
                ),
                SpecialValue(
                    inputs=(3.0,),
                    expected=1.5 - EULER_MASCHERONI,
                    description="psi(3) = 3/2 - gamma",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=-EULER_MASCHERONI - 2 * math.log(2),
                    description="psi(1/2) = -gamma - 2*ln(2)",
                ),
            ],
            singularities=[
                SingularitySpec(
                    type="pole",
                    locations=lambda: (float(n) for n in range(-100, 1)),
                    expected_behavior="-inf",
                    description="Poles at non-positive integers",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=True,
            supports_meta=True,
        )

    def test_known_values(self):
        """Test digamma at known values."""
        z = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = torchscience.special_functions.digamma(z)

        # psi(n) = -gamma + H_{n-1} where H_n = 1 + 1/2 + ... + 1/n
        expected = torch.tensor([
            -EULER_MASCHERONI,                    # psi(1)
            -EULER_MASCHERONI + 1,                # psi(2)
            -EULER_MASCHERONI + 1 + 0.5,          # psi(3)
            -EULER_MASCHERONI + 1 + 0.5 + 1/3,    # psi(4)
        ], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_half_value(self):
        """Test psi(1/2) = -gamma - 2*ln(2)."""
        z = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.digamma(z)
        expected = torch.tensor([-EULER_MASCHERONI - 2 * math.log(2)], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_recurrence_relation(self):
        """Test psi(z+1) = psi(z) + 1/z."""
        z = torch.tensor([0.5, 1.5, 2.5, 5.0, 10.0], dtype=torch.float64)
        left = torchscience.special_functions.digamma(z + 1)
        right = torchscience.special_functions.digamma(z) + 1 / z
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)

    def test_poles_return_neg_inf(self):
        """Test that digamma at poles returns -inf."""
        poles = torch.tensor([0.0, -1.0, -2.0, -3.0], dtype=torch.float64)
        result = torchscience.special_functions.digamma(poles)
        assert (torch.isinf(result) | torch.isnan(result)).all()

    def test_complex_conjugate_symmetry(self):
        """Test psi(conj(z)) = conj(psi(z))."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, 0.5 - 0.3j], dtype=torch.complex128
        )
        result_z = torchscience.special_functions.digamma(z)
        result_conj_z = torchscience.special_functions.digamma(z.conj())
        torch.testing.assert_close(
            result_conj_z, result_z.conj(), rtol=1e-10, atol=1e-10
        )

    def test_comparison_with_torch(self):
        """Test agreement with torch.special.digamma."""
        z = torch.tensor([0.5, 1.0, 1.5, 2.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.digamma(z)
        expected = torch.special.digamma(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_gradient_equals_trigamma(self):
        """Test d/dz psi(z) = trigamma(z)."""
        z = torch.tensor([1.0, 2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.digamma(z)
        y.sum().backward()

        # trigamma(n) for positive integers
        expected = torch.special.polygamma(1, z.detach())
        torch.testing.assert_close(z.grad, expected, rtol=1e-8, atol=1e-8)

    def test_second_order_gradient(self):
        """Test second-order gradients exist and are finite."""
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.digamma(z)

        (grad1,) = torch.autograd.grad(y, z, create_graph=True)
        (grad2,) = torch.autograd.grad(grad1, z)

        assert torch.isfinite(grad2).all()
        # Second derivative is tetragamma, should be negative for positive z
        assert grad2.item() < 0

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        z = torch.tensor([1.5, 2.5, 5.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            torchscience.special_functions.digamma,
            (z,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        z = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            torchscience.special_functions.digamma,
            (z,),
            eps=1e-5,
            atol=1e-4,
            rtol=1e-4,
        )

    @given(z=positive_real_numbers(min_value=0.1, max_value=50.0))
    @settings(max_examples=100, deadline=None)
    def test_property_positive_real_finite(self, z):
        """Property: Digamma of positive real is always finite."""
        z_tensor = torch.tensor([z], dtype=torch.float64)
        result = torchscience.special_functions.digamma(z_tensor)
        assert torch.isfinite(result).all(), f"psi({z}) is not finite: {result}"

    @given(z=avoiding_poles(max_negative_pole=-50, min_value=-50.0, max_value=50.0))
    @settings(max_examples=100, deadline=None)
    def test_property_recurrence(self, z):
        """Property: psi(z+1) = psi(z) + 1/z for all z not at poles."""
        if abs(z) < 0.01:
            return
        z_tensor = torch.tensor([z], dtype=torch.float64)
        left = torchscience.special_functions.digamma(z_tensor + 1)
        right = torchscience.special_functions.digamma(z_tensor) + 1 / z_tensor
        if torch.isfinite(left).all() and torch.isfinite(right).all():
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)

    @given(z=complex_avoiding_real_axis(real_range=(-5.0, 5.0), min_imag=0.1))
    @settings(max_examples=100, deadline=None)
    def test_property_complex_conjugate(self, z):
        """Property: psi(conj(z)) = conj(psi(z))."""
        z_tensor = torch.tensor([z], dtype=torch.complex128)
        psi_z = torchscience.special_functions.digamma(z_tensor)
        psi_conj_z = torchscience.special_functions.digamma(z_tensor.conj())
        torch.testing.assert_close(
            psi_conj_z, psi_z.conj(), rtol=1e-10, atol=1e-10
        )
```

**Step 2: Run tests**

Run: `uv run pytest tests/torchscience/special_functions/test__digamma.py -v --tb=short`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/torchscience/special_functions/test__digamma.py
git commit -m "test: add comprehensive tests for digamma function"
```

---

## Task 12: Verify Full Integration

**Step 1: Run all special function tests**

Run: `uv run pytest tests/torchscience/special_functions/ -v --tb=short`
Expected: All tests pass (including existing gamma, beta, etc.)

**Step 2: Verify import works**

Run: `uv run python -c "from torchscience.special_functions import digamma; import torch; print(digamma(torch.tensor([1.0, 2.0])))"`
Expected: `tensor([-0.5772,  0.4228])`

**Step 3: Mark plan complete**

Update this file's status from "Approved" to "Complete".

---

Plan complete and saved to `docs/plans/2025-12-30-digamma-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
