# torchscience.polynomial Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement differentiable polynomial arithmetic for PyTorch tensors with full autograd support.

**Architecture:** Pure Python implementation using tensorclass data container. Ascending coefficient order (NumPy convention), batch dimensions first. All operations as pure functions for vmap/jacrev compatibility.

**Tech Stack:** PyTorch, tensordict (tensorclass), numpy (test comparisons only)

---

## Phase 1: Module Structure and Exceptions

### Task 1.1: Create module directory structure

**Files:**
- Create: `src/torchscience/polynomial/__init__.py`
- Create: `src/torchscience/polynomial/_exceptions.py`
- Create: `tests/torchscience/polynomial/__init__.py`

**Step 1: Create the polynomial module directory**

```bash
mkdir -p src/torchscience/polynomial
mkdir -p tests/torchscience/polynomial
```

**Step 2: Write the exceptions module**

```python
# src/torchscience/polynomial/_exceptions.py
"""Exceptions for the polynomial module."""


class PolynomialError(Exception):
    """Base exception for polynomial operations."""

    pass


class DegreeError(PolynomialError):
    """Raised when degree is invalid for operation."""

    pass
```

**Step 3: Write the test __init__.py**

```python
# tests/torchscience/polynomial/__init__.py
"""Tests for torchscience.polynomial module."""
```

**Step 4: Write the module __init__.py (minimal)**

```python
# src/torchscience/polynomial/__init__.py
"""Differentiable polynomial arithmetic for PyTorch tensors."""

from torchscience.polynomial._exceptions import (
    DegreeError,
    PolynomialError,
)

__all__ = [
    "PolynomialError",
    "DegreeError",
]
```

**Step 5: Verify imports work**

Run: `uv run python -c "from torchscience.polynomial import PolynomialError, DegreeError; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add src/torchscience/polynomial/ tests/torchscience/polynomial/
git commit -m "feat(polynomial): add module structure and exceptions"
```

---

### Task 1.2: Write failing test for polynomial constructor

**Files:**
- Create: `tests/torchscience/polynomial/test__polynomial.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/polynomial/test__polynomial.py
"""Tests for polynomial core operations."""

import pytest
import torch


class TestPolynomialConstructor:
    def test_create_simple_polynomial(self):
        """Test creating a simple polynomial 1 + 2x + 3x^2."""
        from torchscience.polynomial import Polynomial, polynomial

        coeffs = torch.tensor([1.0, 2.0, 3.0])
        p = polynomial(coeffs)

        assert isinstance(p, Polynomial)
        torch.testing.assert_close(p.coeffs, coeffs)

    def test_create_constant_polynomial(self):
        """Test creating a constant polynomial."""
        from torchscience.polynomial import polynomial

        coeffs = torch.tensor([5.0])
        p = polynomial(coeffs)

        assert p.coeffs.shape == (1,)
        torch.testing.assert_close(p.coeffs, coeffs)

    def test_empty_coefficients_raises(self):
        """Test that empty coefficients raise PolynomialError."""
        from torchscience.polynomial import PolynomialError, polynomial

        with pytest.raises(PolynomialError):
            polynomial(torch.tensor([]))

    def test_batched_polynomial(self):
        """Test creating a batch of polynomials."""
        from torchscience.polynomial import polynomial

        # Batch of 2 polynomials, each degree 2
        coeffs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        p = polynomial(coeffs)

        assert p.coeffs.shape == (2, 3)

    def test_preserves_dtype(self):
        """Test that polynomial preserves input dtype."""
        from torchscience.polynomial import polynomial

        coeffs = torch.tensor([1.0, 2.0], dtype=torch.float64)
        p = polynomial(coeffs)

        assert p.coeffs.dtype == torch.float64

    def test_preserves_device(self):
        """Test that polynomial preserves input device."""
        from torchscience.polynomial import polynomial

        coeffs = torch.tensor([1.0, 2.0])
        p = polynomial(coeffs)

        assert p.coeffs.device == coeffs.device
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialConstructor -v`
Expected: FAIL with "cannot import name 'Polynomial'"

**Step 3: Commit failing test**

```bash
git add tests/torchscience/polynomial/test__polynomial.py
git commit -m "test(polynomial): add failing tests for polynomial constructor"
```

---

### Task 1.3: Implement Polynomial tensorclass and constructor

**Files:**
- Create: `src/torchscience/polynomial/_polynomial.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Write the Polynomial tensorclass and constructor**

```python
# src/torchscience/polynomial/_polynomial.py
"""Polynomial tensorclass and core operations."""

from __future__ import annotations

from typing import Union

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._exceptions import PolynomialError


@tensorclass
class Polynomial:
    """Polynomial in power basis with ascending coefficients.

    Represents p(x) = coeffs[..., 0] + coeffs[..., 1]*x + coeffs[..., 2]*x^2 + ...

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., i] is the coefficient of x^i.
        Batch dimensions come first, coefficient dimension last.
    """

    coeffs: Tensor

    def __add__(self, other: Polynomial) -> Polynomial:
        return polynomial_add(self, other)

    def __radd__(self, other: Polynomial) -> Polynomial:
        return polynomial_add(other, self)

    def __sub__(self, other: Polynomial) -> Polynomial:
        return polynomial_subtract(self, other)

    def __rsub__(self, other: Polynomial) -> Polynomial:
        return polynomial_subtract(other, self)

    def __mul__(self, other: Union[Polynomial, Tensor]) -> Polynomial:
        if isinstance(other, Polynomial):
            return polynomial_multiply(self, other)
        return polynomial_scale(self, other)

    def __rmul__(self, other: Union[Polynomial, Tensor]) -> Polynomial:
        if isinstance(other, Polynomial):
            return polynomial_multiply(other, self)
        return polynomial_scale(self, other)

    def __neg__(self) -> Polynomial:
        return polynomial_negate(self)

    def __call__(self, x: Tensor) -> Tensor:
        return polynomial_evaluate(self, x)


def polynomial(coeffs: Tensor) -> Polynomial:
    """Create polynomial from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        Must have at least one coefficient.

    Returns
    -------
    Polynomial
        Polynomial instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError("Polynomial must have at least one coefficient")

    return Polynomial(coeffs=coeffs, batch_size=coeffs.shape[:-1])


# Forward declarations for operator overloading - implemented in later tasks
def polynomial_add(p: Polynomial, q: Polynomial) -> Polynomial:
    raise NotImplementedError("polynomial_add not yet implemented")


def polynomial_subtract(p: Polynomial, q: Polynomial) -> Polynomial:
    raise NotImplementedError("polynomial_subtract not yet implemented")


def polynomial_multiply(p: Polynomial, q: Polynomial) -> Polynomial:
    raise NotImplementedError("polynomial_multiply not yet implemented")


def polynomial_scale(p: Polynomial, c: Tensor) -> Polynomial:
    raise NotImplementedError("polynomial_scale not yet implemented")


def polynomial_negate(p: Polynomial) -> Polynomial:
    raise NotImplementedError("polynomial_negate not yet implemented")


def polynomial_evaluate(p: Polynomial, x: Tensor) -> Tensor:
    raise NotImplementedError("polynomial_evaluate not yet implemented")
```

**Step 2: Update __init__.py to export Polynomial and polynomial**

```python
# src/torchscience/polynomial/__init__.py
"""Differentiable polynomial arithmetic for PyTorch tensors."""

from torchscience.polynomial._exceptions import (
    DegreeError,
    PolynomialError,
)
from torchscience.polynomial._polynomial import (
    Polynomial,
    polynomial,
)

__all__ = [
    # Data types
    "Polynomial",
    # Constructors
    "polynomial",
    # Exceptions
    "PolynomialError",
    "DegreeError",
]
```

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialConstructor -v`
Expected: All 6 tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement Polynomial tensorclass and constructor"
```

---

## Phase 2: Core Arithmetic

### Task 2.1: Write failing tests for polynomial_negate and polynomial_scale

**Files:**
- Modify: `tests/torchscience/polynomial/test__polynomial.py`

**Step 1: Add failing tests**

```python
# Add to tests/torchscience/polynomial/test__polynomial.py

class TestPolynomialNegate:
    def test_negate_simple(self):
        """Test negating a simple polynomial."""
        from torchscience.polynomial import polynomial, polynomial_negate

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        neg_p = polynomial_negate(p)

        torch.testing.assert_close(neg_p.coeffs, torch.tensor([-1.0, -2.0, -3.0]))

    def test_negate_operator(self):
        """Test negation via operator."""
        from torchscience.polynomial import polynomial

        p = polynomial(torch.tensor([1.0, 2.0]))
        neg_p = -p

        torch.testing.assert_close(neg_p.coeffs, torch.tensor([-1.0, -2.0]))

    def test_negate_batched(self):
        """Test negating batched polynomials."""
        from torchscience.polynomial import polynomial, polynomial_negate

        coeffs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        p = polynomial(coeffs)
        neg_p = polynomial_negate(p)

        torch.testing.assert_close(neg_p.coeffs, -coeffs)


class TestPolynomialScale:
    def test_scale_simple(self):
        """Test scaling a polynomial by a scalar."""
        from torchscience.polynomial import polynomial, polynomial_scale

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        scaled = polynomial_scale(p, torch.tensor(2.0))

        torch.testing.assert_close(scaled.coeffs, torch.tensor([2.0, 4.0, 6.0]))

    def test_scale_operator(self):
        """Test scaling via operator."""
        from torchscience.polynomial import polynomial

        p = polynomial(torch.tensor([1.0, 2.0]))
        scaled = p * torch.tensor(3.0)

        torch.testing.assert_close(scaled.coeffs, torch.tensor([3.0, 6.0]))

    def test_scale_operator_reverse(self):
        """Test scaling via reverse operator."""
        from torchscience.polynomial import polynomial

        p = polynomial(torch.tensor([1.0, 2.0]))
        scaled = torch.tensor(3.0) * p

        torch.testing.assert_close(scaled.coeffs, torch.tensor([3.0, 6.0]))

    def test_scale_batched(self):
        """Test scaling batched polynomials."""
        from torchscience.polynomial import polynomial, polynomial_scale

        coeffs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        p = polynomial(coeffs)
        scaled = polynomial_scale(p, torch.tensor(2.0))

        torch.testing.assert_close(scaled.coeffs, 2.0 * coeffs)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialNegate -v`
Expected: FAIL with "NotImplementedError"

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/polynomial/test__polynomial.py
git commit -m "test(polynomial): add failing tests for negate and scale"
```

---

### Task 2.2: Implement polynomial_negate and polynomial_scale

**Files:**
- Modify: `src/torchscience/polynomial/_polynomial.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Replace stub implementations**

In `src/torchscience/polynomial/_polynomial.py`, replace the stub functions:

```python
def polynomial_negate(p: Polynomial) -> Polynomial:
    """Negate polynomial.

    Parameters
    ----------
    p : Polynomial
        Input polynomial.

    Returns
    -------
    Polynomial
        Negated polynomial -p.
    """
    return Polynomial(coeffs=-p.coeffs, batch_size=p.batch_size)


def polynomial_scale(p: Polynomial, c: Tensor) -> Polynomial:
    """Multiply polynomial by scalar(s).

    Parameters
    ----------
    p : Polynomial
        Polynomial to scale.
    c : Tensor
        Scalar(s), broadcasts with batch dimensions.

    Returns
    -------
    Polynomial
        Scaled polynomial c * p.
    """
    # Expand c to broadcast with coefficients
    scaled_coeffs = p.coeffs * c.unsqueeze(-1) if c.dim() > 0 else p.coeffs * c
    return Polynomial(coeffs=scaled_coeffs, batch_size=p.batch_size)
```

**Step 2: Update __init__.py exports**

```python
# Add to imports in src/torchscience/polynomial/__init__.py
from torchscience.polynomial._polynomial import (
    Polynomial,
    polynomial,
    polynomial_negate,
    polynomial_scale,
)

# Add to __all__
__all__ = [
    # Data types
    "Polynomial",
    # Constructors
    "polynomial",
    # Arithmetic
    "polynomial_negate",
    "polynomial_scale",
    # Exceptions
    "PolynomialError",
    "DegreeError",
]
```

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialNegate tests/torchscience/polynomial/test__polynomial.py::TestPolynomialScale -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement polynomial_negate and polynomial_scale"
```

---

### Task 2.3: Write failing tests for polynomial_add and polynomial_subtract

**Files:**
- Modify: `tests/torchscience/polynomial/test__polynomial.py`

**Step 1: Add failing tests**

```python
# Add to tests/torchscience/polynomial/test__polynomial.py

class TestPolynomialAdd:
    def test_add_same_degree(self):
        """Test adding polynomials of same degree."""
        from torchscience.polynomial import polynomial, polynomial_add

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([4.0, 5.0, 6.0]))
        result = polynomial_add(p, q)

        torch.testing.assert_close(result.coeffs, torch.tensor([5.0, 7.0, 9.0]))

    def test_add_different_degree(self):
        """Test adding polynomials of different degrees."""
        from torchscience.polynomial import polynomial, polynomial_add

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
        q = polynomial(torch.tensor([4.0, 5.0]))  # 4 + 5x
        result = polynomial_add(p, q)

        torch.testing.assert_close(result.coeffs, torch.tensor([5.0, 7.0, 3.0]))

    def test_add_operator(self):
        """Test addition via operator."""
        from torchscience.polynomial import polynomial

        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([3.0, 4.0]))
        result = p + q

        torch.testing.assert_close(result.coeffs, torch.tensor([4.0, 6.0]))

    def test_add_batched(self):
        """Test adding batched polynomials."""
        from torchscience.polynomial import polynomial, polynomial_add

        p = polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        q = polynomial(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        result = polynomial_add(p, q)

        expected = torch.tensor([[6.0, 8.0], [10.0, 12.0]])
        torch.testing.assert_close(result.coeffs, expected)


class TestPolynomialSubtract:
    def test_subtract_same_degree(self):
        """Test subtracting polynomials of same degree."""
        from torchscience.polynomial import polynomial, polynomial_subtract

        p = polynomial(torch.tensor([5.0, 7.0, 9.0]))
        q = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        result = polynomial_subtract(p, q)

        torch.testing.assert_close(result.coeffs, torch.tensor([4.0, 5.0, 6.0]))

    def test_subtract_different_degree(self):
        """Test subtracting polynomials of different degrees."""
        from torchscience.polynomial import polynomial, polynomial_subtract

        p = polynomial(torch.tensor([5.0, 7.0, 9.0]))  # 5 + 7x + 9x^2
        q = polynomial(torch.tensor([1.0, 2.0]))  # 1 + 2x
        result = polynomial_subtract(p, q)

        torch.testing.assert_close(result.coeffs, torch.tensor([4.0, 5.0, 9.0]))

    def test_subtract_operator(self):
        """Test subtraction via operator."""
        from torchscience.polynomial import polynomial

        p = polynomial(torch.tensor([5.0, 6.0]))
        q = polynomial(torch.tensor([1.0, 2.0]))
        result = p - q

        torch.testing.assert_close(result.coeffs, torch.tensor([4.0, 4.0]))
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialAdd -v`
Expected: FAIL with "NotImplementedError"

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/polynomial/test__polynomial.py
git commit -m "test(polynomial): add failing tests for add and subtract"
```

---

### Task 2.4: Implement polynomial_add and polynomial_subtract

**Files:**
- Modify: `src/torchscience/polynomial/_polynomial.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Replace stub implementations**

```python
def polynomial_add(p: Polynomial, q: Polynomial) -> Polynomial:
    """Add two polynomials.

    Broadcasts batch dimensions. Result degree is max(deg(p), deg(q)).

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to add.

    Returns
    -------
    Polynomial
        Sum p + q.
    """
    p_coeffs = p.coeffs
    q_coeffs = q.coeffs

    # Pad shorter polynomial with zeros
    p_deg = p_coeffs.shape[-1]
    q_deg = q_coeffs.shape[-1]

    if p_deg < q_deg:
        pad_shape = list(p_coeffs.shape)
        pad_shape[-1] = q_deg - p_deg
        p_coeffs = torch.cat([p_coeffs, torch.zeros(pad_shape, dtype=p_coeffs.dtype, device=p_coeffs.device)], dim=-1)
    elif q_deg < p_deg:
        pad_shape = list(q_coeffs.shape)
        pad_shape[-1] = p_deg - q_deg
        q_coeffs = torch.cat([q_coeffs, torch.zeros(pad_shape, dtype=q_coeffs.dtype, device=q_coeffs.device)], dim=-1)

    result_coeffs = p_coeffs + q_coeffs
    return Polynomial(coeffs=result_coeffs, batch_size=result_coeffs.shape[:-1])


def polynomial_subtract(p: Polynomial, q: Polynomial) -> Polynomial:
    """Subtract q from p.

    Broadcasts batch dimensions. Result degree is max(deg(p), deg(q)).

    Parameters
    ----------
    p, q : Polynomial
        Polynomials.

    Returns
    -------
    Polynomial
        Difference p - q.
    """
    return polynomial_add(p, polynomial_negate(q))
```

**Step 2: Update __init__.py exports**

Add `polynomial_add` and `polynomial_subtract` to imports and `__all__`.

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialAdd tests/torchscience/polynomial/test__polynomial.py::TestPolynomialSubtract -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement polynomial_add and polynomial_subtract"
```

---

### Task 2.5: Write failing tests for polynomial_multiply

**Files:**
- Modify: `tests/torchscience/polynomial/test__polynomial.py`

**Step 1: Add failing tests**

```python
# Add to tests/torchscience/polynomial/test__polynomial.py

class TestPolynomialMultiply:
    def test_multiply_simple(self):
        """Test multiplying two simple polynomials."""
        from torchscience.polynomial import polynomial, polynomial_multiply

        # (1 + 2x) * (3 + 4x) = 3 + 4x + 6x + 8x^2 = 3 + 10x + 8x^2
        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([3.0, 4.0]))
        result = polynomial_multiply(p, q)

        torch.testing.assert_close(result.coeffs, torch.tensor([3.0, 10.0, 8.0]))

    def test_multiply_by_constant(self):
        """Test multiplying by a constant polynomial."""
        from torchscience.polynomial import polynomial, polynomial_multiply

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([2.0]))
        result = polynomial_multiply(p, q)

        torch.testing.assert_close(result.coeffs, torch.tensor([2.0, 4.0, 6.0]))

    def test_multiply_operator(self):
        """Test multiplication via operator."""
        from torchscience.polynomial import polynomial

        p = polynomial(torch.tensor([1.0, 1.0]))  # 1 + x
        q = polynomial(torch.tensor([1.0, 1.0]))  # 1 + x
        result = p * q  # (1+x)^2 = 1 + 2x + x^2

        torch.testing.assert_close(result.coeffs, torch.tensor([1.0, 2.0, 1.0]))

    def test_multiply_batched(self):
        """Test multiplying batched polynomials."""
        from torchscience.polynomial import polynomial, polynomial_multiply

        # Batch of 2 polynomials
        p = polynomial(torch.tensor([[1.0, 2.0], [1.0, 0.0]]))  # [1+2x, 1]
        q = polynomial(torch.tensor([[1.0, 1.0], [2.0, 3.0]]))  # [1+x, 2+3x]
        result = polynomial_multiply(p, q)

        # [1+2x][1+x] = 1 + 3x + 2x^2
        # [1][2+3x] = 2 + 3x
        expected = torch.tensor([[1.0, 3.0, 2.0], [2.0, 3.0, 0.0]])
        torch.testing.assert_close(result.coeffs, expected)

    def test_multiply_result_degree(self):
        """Test that result has correct degree."""
        from torchscience.polynomial import polynomial, polynomial_multiply

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # degree 2
        q = polynomial(torch.tensor([4.0, 5.0]))  # degree 1
        result = polynomial_multiply(p, q)

        # degree should be 2 + 1 = 3, so 4 coefficients
        assert result.coeffs.shape[-1] == 4
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialMultiply -v`
Expected: FAIL with "NotImplementedError"

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/polynomial/test__polynomial.py
git commit -m "test(polynomial): add failing tests for polynomial_multiply"
```

---

### Task 2.6: Implement polynomial_multiply

**Files:**
- Modify: `src/torchscience/polynomial/_polynomial.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Replace stub implementation**

```python
def polynomial_multiply(p: Polynomial, q: Polynomial) -> Polynomial:
    """Multiply two polynomials.

    Computes convolution of coefficients. Result degree is deg(p) + deg(q).

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to multiply.

    Returns
    -------
    Polynomial
        Product p * q.
    """
    p_coeffs = p.coeffs
    q_coeffs = q.coeffs

    p_deg = p_coeffs.shape[-1]
    q_deg = q_coeffs.shape[-1]
    result_deg = p_deg + q_deg - 1

    # Handle batch dimensions
    batch_shape = torch.broadcast_shapes(p_coeffs.shape[:-1], q_coeffs.shape[:-1])

    # Broadcast to common batch shape
    p_coeffs = p_coeffs.broadcast_to(*batch_shape, p_deg)
    q_coeffs = q_coeffs.broadcast_to(*batch_shape, q_deg)

    # Compute convolution (polynomial multiplication)
    # result[k] = sum_{i+j=k} p[i] * q[j]
    result_coeffs = torch.zeros(*batch_shape, result_deg, dtype=p_coeffs.dtype, device=p_coeffs.device)

    for i in range(p_deg):
        for j in range(q_deg):
            result_coeffs[..., i + j] = result_coeffs[..., i + j] + p_coeffs[..., i] * q_coeffs[..., j]

    return Polynomial(coeffs=result_coeffs, batch_size=batch_shape)
```

**Step 2: Update __init__.py exports**

Add `polynomial_multiply` to imports and `__all__`.

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialMultiply -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement polynomial_multiply"
```

---

### Task 2.7: Write failing tests for polynomial_degree

**Files:**
- Modify: `tests/torchscience/polynomial/test__polynomial.py`

**Step 1: Add failing tests**

```python
# Add to tests/torchscience/polynomial/test__polynomial.py

class TestPolynomialDegree:
    def test_degree_simple(self):
        """Test degree of simple polynomial."""
        from torchscience.polynomial import polynomial, polynomial_degree

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # degree 2
        deg = polynomial_degree(p)

        assert deg == 2

    def test_degree_constant(self):
        """Test degree of constant polynomial."""
        from torchscience.polynomial import polynomial, polynomial_degree

        p = polynomial(torch.tensor([5.0]))  # degree 0
        deg = polynomial_degree(p)

        assert deg == 0

    def test_degree_batched(self):
        """Test degree of batched polynomials."""
        from torchscience.polynomial import polynomial, polynomial_degree

        p = polynomial(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        deg = polynomial_degree(p)

        # All have same degree since same shape
        torch.testing.assert_close(deg, torch.tensor([2, 2]))
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialDegree -v`
Expected: FAIL with "polynomial_degree" not found

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/polynomial/test__polynomial.py
git commit -m "test(polynomial): add failing tests for polynomial_degree"
```

---

### Task 2.8: Implement polynomial_degree

**Files:**
- Modify: `src/torchscience/polynomial/_polynomial.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Add implementation**

```python
def polynomial_degree(p: Polynomial) -> Tensor:
    """Return degree of polynomial(s).

    Parameters
    ----------
    p : Polynomial
        Input polynomial.

    Returns
    -------
    Tensor
        Degree, shape matches batch dimensions.
        Degree is N-1 where N is the number of coefficients.
    """
    n_coeffs = p.coeffs.shape[-1]
    degree = n_coeffs - 1

    # Return scalar for unbatched, tensor for batched
    if p.batch_size == torch.Size([]):
        return torch.tensor(degree, device=p.coeffs.device)
    else:
        return torch.full(p.batch_size, degree, device=p.coeffs.device)
```

**Step 2: Update __init__.py exports**

Add `polynomial_degree` to imports and `__all__`.

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialDegree -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement polynomial_degree"
```

---

## Phase 3: Evaluation and Calculus

### Task 3.1: Write failing tests for polynomial_evaluate

**Files:**
- Modify: `tests/torchscience/polynomial/test__polynomial.py`

**Step 1: Add failing tests**

```python
# Add to tests/torchscience/polynomial/test__polynomial.py

class TestPolynomialEvaluate:
    def test_evaluate_simple(self):
        """Test evaluating a simple polynomial."""
        from torchscience.polynomial import polynomial, polynomial_evaluate

        # 1 + 2x + 3x^2 at x=2: 1 + 4 + 12 = 17
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        result = polynomial_evaluate(p, torch.tensor(2.0))

        torch.testing.assert_close(result, torch.tensor(17.0))

    def test_evaluate_at_zero(self):
        """Test evaluating at x=0 returns constant term."""
        from torchscience.polynomial import polynomial, polynomial_evaluate

        p = polynomial(torch.tensor([5.0, 2.0, 3.0]))
        result = polynomial_evaluate(p, torch.tensor(0.0))

        torch.testing.assert_close(result, torch.tensor(5.0))

    def test_evaluate_multiple_points(self):
        """Test evaluating at multiple points."""
        from torchscience.polynomial import polynomial, polynomial_evaluate

        # 1 + 2x + 3x^2
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.tensor([0.0, 1.0, 2.0])
        result = polynomial_evaluate(p, x)

        # x=0: 1, x=1: 1+2+3=6, x=2: 1+4+12=17
        torch.testing.assert_close(result, torch.tensor([1.0, 6.0, 17.0]))

    def test_evaluate_call_operator(self):
        """Test evaluation via __call__."""
        from torchscience.polynomial import polynomial

        p = polynomial(torch.tensor([1.0, 2.0]))  # 1 + 2x
        result = p(torch.tensor(3.0))

        torch.testing.assert_close(result, torch.tensor(7.0))

    def test_evaluate_batched(self):
        """Test evaluating batched polynomials."""
        from torchscience.polynomial import polynomial, polynomial_evaluate

        # Batch of 2 polynomials
        p = polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        x = torch.tensor([1.0, 1.0])  # Evaluate each at x=1
        result = polynomial_evaluate(p, x)

        # [1+2, 3+4] = [3, 7]
        torch.testing.assert_close(result, torch.tensor([3.0, 7.0]))

    def test_evaluate_gradcheck(self):
        """Test autograd through evaluation."""
        from torchscience.polynomial import polynomial, polynomial_evaluate

        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([0.5, 1.5], dtype=torch.float64, requires_grad=True)

        def fn(c, x):
            p = polynomial(c)
            return polynomial_evaluate(p, x)

        torch.autograd.gradcheck(fn, (coeffs, x))
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialEvaluate -v`
Expected: FAIL with "NotImplementedError"

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/polynomial/test__polynomial.py
git commit -m "test(polynomial): add failing tests for polynomial_evaluate"
```

---

### Task 3.2: Implement polynomial_evaluate

**Files:**
- Modify: `src/torchscience/polynomial/_polynomial.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Replace stub implementation**

```python
def polynomial_evaluate(p: Polynomial, x: Tensor) -> Tensor:
    """Evaluate polynomial at points using Horner's method.

    Parameters
    ----------
    p : Polynomial
        Polynomial with coefficients shape (..., N).
    x : Tensor
        Evaluation points, broadcasts with p's batch dimensions.

    Returns
    -------
    Tensor
        Values p(x).
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    # Horner's method: result = c[n-1]; for i in range(n-2, -1, -1): result = result * x + c[i]
    # Start with highest degree coefficient
    result = coeffs[..., n - 1]

    # Expand x to broadcast with batch dimensions
    for _ in range(coeffs.dim() - 1):
        if x.dim() < coeffs.dim() - 1:
            x = x.unsqueeze(0)

    for i in range(n - 2, -1, -1):
        result = result * x + coeffs[..., i]

    return result
```

**Step 2: Update __init__.py exports**

Add `polynomial_evaluate` to imports and `__all__`.

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialEvaluate -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement polynomial_evaluate with Horner's method"
```

---

### Task 3.3: Write failing tests for polynomial_derivative

**Files:**
- Modify: `tests/torchscience/polynomial/test__polynomial.py`

**Step 1: Add failing tests**

```python
# Add to tests/torchscience/polynomial/test__polynomial.py

class TestPolynomialDerivative:
    def test_derivative_simple(self):
        """Test derivative of simple polynomial."""
        from torchscience.polynomial import polynomial, polynomial_derivative

        # d/dx(1 + 2x + 3x^2) = 2 + 6x
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        dp = polynomial_derivative(p)

        torch.testing.assert_close(dp.coeffs, torch.tensor([2.0, 6.0]))

    def test_derivative_constant(self):
        """Test derivative of constant is zero polynomial."""
        from torchscience.polynomial import polynomial, polynomial_derivative

        p = polynomial(torch.tensor([5.0]))
        dp = polynomial_derivative(p)

        torch.testing.assert_close(dp.coeffs, torch.tensor([0.0]))

    def test_derivative_linear(self):
        """Test derivative of linear is constant."""
        from torchscience.polynomial import polynomial, polynomial_derivative

        p = polynomial(torch.tensor([3.0, 7.0]))  # 3 + 7x
        dp = polynomial_derivative(p)

        torch.testing.assert_close(dp.coeffs, torch.tensor([7.0]))

    def test_derivative_second_order(self):
        """Test second derivative."""
        from torchscience.polynomial import polynomial, polynomial_derivative

        # d^2/dx^2(1 + 2x + 3x^2 + 4x^3) = 6 + 24x
        p = polynomial(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        d2p = polynomial_derivative(p, order=2)

        torch.testing.assert_close(d2p.coeffs, torch.tensor([6.0, 24.0]))

    def test_derivative_batched(self):
        """Test derivative of batched polynomials."""
        from torchscience.polynomial import polynomial, polynomial_derivative

        p = polynomial(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        dp = polynomial_derivative(p)

        expected = torch.tensor([[2.0, 6.0], [5.0, 12.0]])
        torch.testing.assert_close(dp.coeffs, expected)

    def test_derivative_matches_evaluate_gradient(self):
        """Test that derivative matches autograd through evaluate."""
        from torchscience.polynomial import polynomial, polynomial_derivative, polynomial_evaluate

        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0])
        p = polynomial(coeffs)
        dp = polynomial_derivative(p)

        x = torch.tensor(0.5, requires_grad=True)
        y = polynomial_evaluate(p, x)
        y.backward()

        # Derivative at x should match dp(x)
        torch.testing.assert_close(x.grad, polynomial_evaluate(dp, torch.tensor(0.5)))
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialDerivative -v`
Expected: FAIL with "polynomial_derivative" not found

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/polynomial/test__polynomial.py
git commit -m "test(polynomial): add failing tests for polynomial_derivative"
```

---

### Task 3.4: Implement polynomial_derivative

**Files:**
- Modify: `src/torchscience/polynomial/_polynomial.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Add implementation**

```python
def polynomial_derivative(p: Polynomial, order: int = 1) -> Polynomial:
    """Compute derivative of polynomial.

    Parameters
    ----------
    p : Polynomial
        Input polynomial.
    order : int
        Derivative order (default 1).

    Returns
    -------
    Polynomial
        Derivative d^n p / dx^n. Constant polynomial returns [0.0].
    """
    coeffs = p.coeffs

    for _ in range(order):
        n = coeffs.shape[-1]
        if n <= 1:
            # Derivative of constant is zero
            zero_shape = list(coeffs.shape)
            zero_shape[-1] = 1
            coeffs = torch.zeros(zero_shape, dtype=coeffs.dtype, device=coeffs.device)
        else:
            # new_coeffs[i] = (i+1) * coeffs[i+1]
            indices = torch.arange(1, n, device=coeffs.device, dtype=coeffs.dtype)
            coeffs = coeffs[..., 1:] * indices

    return Polynomial(coeffs=coeffs, batch_size=coeffs.shape[:-1])
```

**Step 2: Update __init__.py exports**

Add `polynomial_derivative` to imports and `__all__`.

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialDerivative -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement polynomial_derivative"
```

---

### Task 3.5: Write failing tests for polynomial_antiderivative

**Files:**
- Modify: `tests/torchscience/polynomial/test__polynomial.py`

**Step 1: Add failing tests**

```python
# Add to tests/torchscience/polynomial/test__polynomial.py

class TestPolynomialAntiderivative:
    def test_antiderivative_simple(self):
        """Test antiderivative of simple polynomial."""
        from torchscience.polynomial import polynomial, polynomial_antiderivative

        # ∫(2 + 6x)dx = C + 2x + 3x^2
        p = polynomial(torch.tensor([2.0, 6.0]))
        ap = polynomial_antiderivative(p)

        torch.testing.assert_close(ap.coeffs, torch.tensor([0.0, 2.0, 3.0]))

    def test_antiderivative_with_constant(self):
        """Test antiderivative with integration constant."""
        from torchscience.polynomial import polynomial, polynomial_antiderivative

        p = polynomial(torch.tensor([2.0, 6.0]))
        ap = polynomial_antiderivative(p, constant=torch.tensor(5.0))

        torch.testing.assert_close(ap.coeffs, torch.tensor([5.0, 2.0, 3.0]))

    def test_antiderivative_constant_poly(self):
        """Test antiderivative of constant."""
        from torchscience.polynomial import polynomial, polynomial_antiderivative

        # ∫3 dx = 3x + C
        p = polynomial(torch.tensor([3.0]))
        ap = polynomial_antiderivative(p)

        torch.testing.assert_close(ap.coeffs, torch.tensor([0.0, 3.0]))

    def test_antiderivative_derivative_roundtrip(self):
        """Test that derivative(antiderivative(p)) = p."""
        from torchscience.polynomial import polynomial, polynomial_antiderivative, polynomial_derivative

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        ap = polynomial_antiderivative(p)
        dap = polynomial_derivative(ap)

        torch.testing.assert_close(dap.coeffs, p.coeffs)

    def test_antiderivative_batched(self):
        """Test antiderivative of batched polynomials."""
        from torchscience.polynomial import polynomial, polynomial_antiderivative

        p = polynomial(torch.tensor([[2.0, 4.0], [3.0, 6.0]]))
        ap = polynomial_antiderivative(p)

        expected = torch.tensor([[0.0, 2.0, 2.0], [0.0, 3.0, 3.0]])
        torch.testing.assert_close(ap.coeffs, expected)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialAntiderivative -v`
Expected: FAIL with "polynomial_antiderivative" not found

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/polynomial/test__polynomial.py
git commit -m "test(polynomial): add failing tests for polynomial_antiderivative"
```

---

### Task 3.6: Implement polynomial_antiderivative

**Files:**
- Modify: `src/torchscience/polynomial/_polynomial.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Add implementation**

```python
def polynomial_antiderivative(p: Polynomial, constant: Tensor = None) -> Polynomial:
    """Compute antiderivative (indefinite integral).

    Parameters
    ----------
    p : Polynomial
        Input polynomial.
    constant : Tensor, optional
        Integration constant (default 0).

    Returns
    -------
    Polynomial
        Antiderivative with given constant term. Degree increases by 1.
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    # new_coeffs[i+1] = coeffs[i] / (i+1)
    indices = torch.arange(1, n + 1, device=coeffs.device, dtype=coeffs.dtype)
    new_coeffs = coeffs / indices

    # Prepend constant term
    if constant is None:
        constant = torch.zeros(coeffs.shape[:-1], dtype=coeffs.dtype, device=coeffs.device)

    constant = constant.unsqueeze(-1)
    result_coeffs = torch.cat([constant, new_coeffs], dim=-1)

    return Polynomial(coeffs=result_coeffs, batch_size=result_coeffs.shape[:-1])
```

**Step 2: Update __init__.py exports**

Add `polynomial_antiderivative` to imports and `__all__`.

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialAntiderivative -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement polynomial_antiderivative"
```

---

### Task 3.7: Write failing tests for polynomial_integral

**Files:**
- Modify: `tests/torchscience/polynomial/test__polynomial.py`

**Step 1: Add failing tests**

```python
# Add to tests/torchscience/polynomial/test__polynomial.py

class TestPolynomialIntegral:
    def test_integral_simple(self):
        """Test definite integral of simple polynomial."""
        from torchscience.polynomial import polynomial, polynomial_integral

        # ∫_0^1 (1 + x^2) dx = [x + x^3/3]_0^1 = 1 + 1/3 = 4/3
        p = polynomial(torch.tensor([1.0, 0.0, 1.0]))
        result = polynomial_integral(p, torch.tensor(0.0), torch.tensor(1.0))

        torch.testing.assert_close(result, torch.tensor(4.0 / 3.0))

    def test_integral_linear(self):
        """Test integral of linear polynomial."""
        from torchscience.polynomial import polynomial, polynomial_integral

        # ∫_0^2 (3 + 2x) dx = [3x + x^2]_0^2 = 6 + 4 = 10
        p = polynomial(torch.tensor([3.0, 2.0]))
        result = polynomial_integral(p, torch.tensor(0.0), torch.tensor(2.0))

        torch.testing.assert_close(result, torch.tensor(10.0))

    def test_integral_negative_bounds(self):
        """Test integral with negative bounds."""
        from torchscience.polynomial import polynomial, polynomial_integral

        # ∫_{-1}^{1} x^2 dx = [x^3/3]_{-1}^{1} = 1/3 - (-1/3) = 2/3
        p = polynomial(torch.tensor([0.0, 0.0, 1.0]))
        result = polynomial_integral(p, torch.tensor(-1.0), torch.tensor(1.0))

        torch.testing.assert_close(result, torch.tensor(2.0 / 3.0))

    def test_integral_batched(self):
        """Test integral of batched polynomials."""
        from torchscience.polynomial import polynomial, polynomial_integral

        # Two constant polynomials
        p = polynomial(torch.tensor([[2.0], [3.0]]))
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([1.0, 2.0])
        result = polynomial_integral(p, a, b)

        # ∫_0^1 2 dx = 2, ∫_0^2 3 dx = 6
        torch.testing.assert_close(result, torch.tensor([2.0, 6.0]))

    def test_integral_gradcheck(self):
        """Test autograd through integral."""
        from torchscience.polynomial import polynomial, polynomial_integral

        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        a = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        def fn(c, a, b):
            p = polynomial(c)
            return polynomial_integral(p, a, b)

        torch.autograd.gradcheck(fn, (coeffs, a, b))
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialIntegral -v`
Expected: FAIL with "polynomial_integral" not found

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/polynomial/test__polynomial.py
git commit -m "test(polynomial): add failing tests for polynomial_integral"
```

---

### Task 3.8: Implement polynomial_integral

**Files:**
- Modify: `src/torchscience/polynomial/_polynomial.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Add implementation**

```python
def polynomial_integral(p: Polynomial, a: Tensor, b: Tensor) -> Tensor:
    """Compute definite integral.

    Parameters
    ----------
    p : Polynomial
        Polynomial to integrate.
    a, b : Tensor
        Integration bounds, broadcast with batch dimensions.

    Returns
    -------
    Tensor
        Definite integral ∫_a^b p(x) dx.
    """
    # Get antiderivative with constant=0
    ap = polynomial_antiderivative(p, constant=None)

    # Evaluate at bounds and subtract
    return polynomial_evaluate(ap, b) - polynomial_evaluate(ap, a)
```

**Step 2: Update __init__.py exports**

Add `polynomial_integral` to imports and `__all__`.

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialIntegral -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement polynomial_integral"
```

---

## Phase 4: Root Finding and Utilities

### Task 4.1: Write failing tests for polynomial_roots

**Files:**
- Create: `tests/torchscience/polynomial/test__roots.py`

**Step 1: Write failing tests**

```python
# tests/torchscience/polynomial/test__roots.py
"""Tests for polynomial root finding."""

import pytest
import torch


class TestPolynomialRoots:
    def test_roots_quadratic(self):
        """Test roots of quadratic polynomial."""
        from torchscience.polynomial import polynomial, polynomial_roots

        # (x-1)(x-2) = x^2 - 3x + 2 = 2 - 3x + x^2
        p = polynomial(torch.tensor([2.0, -3.0, 1.0], dtype=torch.float64))
        roots = polynomial_roots(p)

        # Sort roots for comparison
        roots_sorted = torch.sort(roots.real)[0]
        torch.testing.assert_close(roots_sorted.real, torch.tensor([1.0, 2.0], dtype=torch.float64), atol=1e-10, rtol=1e-10)

    def test_roots_linear(self):
        """Test roots of linear polynomial."""
        from torchscience.polynomial import polynomial, polynomial_roots

        # 2 + 4x = 0 => x = -0.5
        p = polynomial(torch.tensor([2.0, 4.0], dtype=torch.float64))
        roots = polynomial_roots(p)

        torch.testing.assert_close(roots.real, torch.tensor([-0.5], dtype=torch.float64), atol=1e-10, rtol=1e-10)

    def test_roots_complex(self):
        """Test roots that are complex."""
        from torchscience.polynomial import polynomial, polynomial_roots

        # x^2 + 1 = 0 => x = ±i
        p = polynomial(torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64))
        roots = polynomial_roots(p)

        # Should have roots ±i
        assert roots.is_complex()
        roots_imag_sorted = torch.sort(roots.imag)[0]
        torch.testing.assert_close(roots_imag_sorted, torch.tensor([-1.0, 1.0], dtype=torch.float64), atol=1e-10, rtol=1e-10)

    def test_roots_constant_raises(self):
        """Test that constant polynomial raises DegreeError."""
        from torchscience.polynomial import DegreeError, polynomial, polynomial_roots

        p = polynomial(torch.tensor([5.0]))

        with pytest.raises(DegreeError):
            polynomial_roots(p)

    def test_roots_cubic(self):
        """Test roots of cubic polynomial."""
        from torchscience.polynomial import polynomial, polynomial_roots

        # (x-1)(x-2)(x-3) = -6 + 11x - 6x^2 + x^3
        p = polynomial(torch.tensor([-6.0, 11.0, -6.0, 1.0], dtype=torch.float64))
        roots = polynomial_roots(p)

        roots_sorted = torch.sort(roots.real)[0]
        torch.testing.assert_close(roots_sorted.real, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), atol=1e-10, rtol=1e-10)

    def test_roots_batched(self):
        """Test roots of batched polynomials."""
        from torchscience.polynomial import polynomial, polynomial_roots

        # Batch of 2 quadratics
        # (x-1)(x-2) and (x-3)(x-4)
        coeffs = torch.tensor([
            [2.0, -3.0, 1.0],   # x^2 - 3x + 2
            [12.0, -7.0, 1.0],  # x^2 - 7x + 12
        ], dtype=torch.float64)
        p = polynomial(coeffs)
        roots = polynomial_roots(p)

        assert roots.shape == (2, 2)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__roots.py -v`
Expected: FAIL with "polynomial_roots" not found

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/polynomial/test__roots.py
git commit -m "test(polynomial): add failing tests for polynomial_roots"
```

---

### Task 4.2: Implement polynomial_roots

**Files:**
- Create: `src/torchscience/polynomial/_roots.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Write implementation**

```python
# src/torchscience/polynomial/_roots.py
"""Polynomial root finding via companion matrix."""

import torch
from torch import Tensor

from torchscience.polynomial._exceptions import DegreeError
from torchscience.polynomial._polynomial import Polynomial


def polynomial_roots(p: Polynomial) -> Tensor:
    """Find polynomial roots via companion matrix eigenvalues.

    Parameters
    ----------
    p : Polynomial
        Polynomial with coefficients shape (..., N).
        Leading coefficient must be non-zero.

    Returns
    -------
    Tensor
        Complex roots, shape (..., N-1). Always complex dtype.

    Raises
    ------
    DegreeError
        If polynomial is constant (degree 0) or zero polynomial.
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    if n <= 1:
        raise DegreeError("Cannot find roots of constant polynomial")

    degree = n - 1

    # Normalize by leading coefficient
    leading = coeffs[..., -1:]
    normalized = coeffs[..., :-1] / leading

    # Build companion matrix
    # C = [[0, 0, ..., 0, -a_0/a_n],
    #      [1, 0, ..., 0, -a_1/a_n],
    #      [0, 1, ..., 0, -a_2/a_n],
    #      [...                   ],
    #      [0, 0, ..., 1, -a_{n-1}/a_n]]
    batch_shape = coeffs.shape[:-1]

    # Create companion matrix
    companion = torch.zeros(*batch_shape, degree, degree, dtype=coeffs.dtype, device=coeffs.device)

    # Fill subdiagonal with ones
    if degree > 1:
        eye_indices = torch.arange(degree - 1, device=coeffs.device)
        companion[..., eye_indices + 1, eye_indices] = 1.0

    # Fill last column with -normalized coefficients
    companion[..., :, -1] = -normalized

    # Compute eigenvalues
    # Convert to complex for eigvals
    if not companion.is_complex():
        companion = companion.to(torch.complex128 if coeffs.dtype == torch.float64 else torch.complex64)

    roots = torch.linalg.eigvals(companion)

    return roots
```

**Step 2: Update __init__.py exports**

```python
# Add to imports
from torchscience.polynomial._roots import polynomial_roots

# Add to __all__
"polynomial_roots",
```

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__roots.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement polynomial_roots via companion matrix"
```

---

### Task 4.3: Write failing tests for polynomial_from_roots

**Files:**
- Modify: `tests/torchscience/polynomial/test__roots.py`

**Step 1: Add failing tests**

```python
# Add to tests/torchscience/polynomial/test__roots.py

class TestPolynomialFromRoots:
    def test_from_roots_simple(self):
        """Test constructing polynomial from roots."""
        from torchscience.polynomial import polynomial_from_roots

        # (x-1)(x-2) = x^2 - 3x + 2
        roots = torch.tensor([1.0, 2.0])
        p = polynomial_from_roots(roots)

        torch.testing.assert_close(p.coeffs, torch.tensor([2.0, -3.0, 1.0]))

    def test_from_roots_single(self):
        """Test constructing polynomial from single root."""
        from torchscience.polynomial import polynomial_from_roots

        # (x-3) = -3 + x
        roots = torch.tensor([3.0])
        p = polynomial_from_roots(roots)

        torch.testing.assert_close(p.coeffs, torch.tensor([-3.0, 1.0]))

    def test_from_roots_complex(self):
        """Test constructing polynomial from complex roots."""
        from torchscience.polynomial import polynomial_from_roots

        # (x-i)(x+i) = x^2 + 1
        roots = torch.tensor([1j, -1j])
        p = polynomial_from_roots(roots)

        expected = torch.tensor([1.0 + 0j, 0.0 + 0j, 1.0 + 0j])
        torch.testing.assert_close(p.coeffs, expected, atol=1e-10, rtol=1e-10)

    def test_from_roots_roundtrip(self):
        """Test that polynomial_roots(polynomial_from_roots(r)) ≈ r."""
        from torchscience.polynomial import polynomial_from_roots, polynomial_roots

        roots = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        p = polynomial_from_roots(roots)
        recovered = polynomial_roots(p)

        # Sort for comparison
        recovered_sorted = torch.sort(recovered.real)[0]
        torch.testing.assert_close(recovered_sorted, roots, atol=1e-10, rtol=1e-10)

    def test_from_roots_batched(self):
        """Test constructing batched polynomials from roots."""
        from torchscience.polynomial import polynomial_from_roots

        # Batch of 2 sets of roots
        roots = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        p = polynomial_from_roots(roots)

        assert p.coeffs.shape == (2, 3)
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__roots.py::TestPolynomialFromRoots -v`
Expected: FAIL with "polynomial_from_roots" not found

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/polynomial/test__roots.py
git commit -m "test(polynomial): add failing tests for polynomial_from_roots"
```

---

### Task 4.4: Implement polynomial_from_roots

**Files:**
- Modify: `src/torchscience/polynomial/_roots.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Add implementation**

```python
# Add to src/torchscience/polynomial/_roots.py

def polynomial_from_roots(roots: Tensor) -> Polynomial:
    """Construct monic polynomial from its roots.

    Constructs (x - r_0)(x - r_1)...(x - r_{n-1}).

    Parameters
    ----------
    roots : Tensor
        Roots, shape (..., N). Can be complex.

    Returns
    -------
    Polynomial
        Monic polynomial with given roots, shape (..., N+1).
    """
    batch_shape = roots.shape[:-1]
    n = roots.shape[-1]

    # Start with polynomial [1] (constant 1)
    # Then iteratively multiply by (x - r_i)
    # (x - r) in ascending order is [-r, 1]

    coeffs = torch.ones(*batch_shape, 1, dtype=roots.dtype, device=roots.device)

    for i in range(n):
        # Multiply current polynomial by (x - roots[..., i])
        # This is convolving coeffs with [-roots[..., i], 1]
        root_i = roots[..., i:i+1]

        # New coeffs: prepend 0 (shift up by x) then subtract root * old coeffs
        shifted = torch.cat([torch.zeros_like(root_i), coeffs], dim=-1)
        scaled = torch.cat([coeffs * (-root_i), torch.zeros_like(root_i)], dim=-1)
        coeffs = shifted + scaled

    return Polynomial(coeffs=coeffs, batch_size=batch_shape)
```

**Step 2: Update __init__.py exports**

Add `polynomial_from_roots` to imports and `__all__`.

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__roots.py::TestPolynomialFromRoots -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement polynomial_from_roots"
```

---

### Task 4.5: Write failing tests for polynomial_trim and polynomial_equal

**Files:**
- Modify: `tests/torchscience/polynomial/test__polynomial.py`

**Step 1: Add failing tests**

```python
# Add to tests/torchscience/polynomial/test__polynomial.py

class TestPolynomialTrim:
    def test_trim_trailing_zeros(self):
        """Test trimming trailing zeros."""
        from torchscience.polynomial import polynomial, polynomial_trim

        p = polynomial(torch.tensor([1.0, 2.0, 0.0, 0.0]))
        trimmed = polynomial_trim(p)

        torch.testing.assert_close(trimmed.coeffs, torch.tensor([1.0, 2.0]))

    def test_trim_near_zeros(self):
        """Test trimming near-zero coefficients with tolerance."""
        from torchscience.polynomial import polynomial, polynomial_trim

        p = polynomial(torch.tensor([1.0, 2.0, 1e-10, 1e-12]))
        trimmed = polynomial_trim(p, tol=1e-9)

        torch.testing.assert_close(trimmed.coeffs, torch.tensor([1.0, 2.0]))

    def test_trim_keeps_one_coeff(self):
        """Test that trim keeps at least one coefficient."""
        from torchscience.polynomial import polynomial, polynomial_trim

        p = polynomial(torch.tensor([0.0, 0.0, 0.0]))
        trimmed = polynomial_trim(p)

        assert trimmed.coeffs.shape[-1] >= 1

    def test_trim_no_change(self):
        """Test that trim doesn't change polynomial without trailing zeros."""
        from torchscience.polynomial import polynomial, polynomial_trim

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        trimmed = polynomial_trim(p)

        torch.testing.assert_close(trimmed.coeffs, p.coeffs)


class TestPolynomialEqual:
    def test_equal_same(self):
        """Test equality of identical polynomials."""
        from torchscience.polynomial import polynomial, polynomial_equal

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([1.0, 2.0, 3.0]))

        assert polynomial_equal(p, q).all()

    def test_equal_different(self):
        """Test inequality of different polynomials."""
        from torchscience.polynomial import polynomial, polynomial_equal

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([1.0, 2.0, 4.0]))

        assert not polynomial_equal(p, q).all()

    def test_equal_different_degree_with_zeros(self):
        """Test equality when polynomials have trailing zeros."""
        from torchscience.polynomial import polynomial, polynomial_equal

        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([1.0, 2.0, 0.0]))

        # These should be equal as polynomials
        assert polynomial_equal(p, q).all()

    def test_equal_tolerance(self):
        """Test equality with tolerance."""
        from torchscience.polynomial import polynomial, polynomial_equal

        p = polynomial(torch.tensor([1.0, 2.0]))
        q = polynomial(torch.tensor([1.0 + 1e-10, 2.0 - 1e-10]))

        assert polynomial_equal(p, q, tol=1e-8).all()

    def test_equal_batched(self):
        """Test equality of batched polynomials."""
        from torchscience.polynomial import polynomial, polynomial_equal

        p = polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        q = polynomial(torch.tensor([[1.0, 2.0], [3.0, 5.0]]))

        result = polynomial_equal(p, q)
        assert result[0] and not result[1]
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialTrim -v`
Expected: FAIL with "polynomial_trim" not found

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/polynomial/test__polynomial.py
git commit -m "test(polynomial): add failing tests for polynomial_trim and polynomial_equal"
```

---

### Task 4.6: Implement polynomial_trim and polynomial_equal

**Files:**
- Modify: `src/torchscience/polynomial/_polynomial.py`
- Modify: `src/torchscience/polynomial/__init__.py`

**Step 1: Add implementations**

```python
# Add to src/torchscience/polynomial/_polynomial.py

def polynomial_trim(p: Polynomial, tol: float = 0.0) -> Polynomial:
    """Remove trailing near-zero coefficients.

    Parameters
    ----------
    p : Polynomial
        Input polynomial.
    tol : float
        Tolerance for considering coefficient as zero.

    Returns
    -------
    Polynomial
        Trimmed polynomial with at least one coefficient.
    """
    coeffs = p.coeffs

    # For unbatched case
    if coeffs.dim() == 1:
        # Find last non-zero coefficient
        mask = torch.abs(coeffs) > tol
        if not mask.any():
            # All zeros - return single zero
            return Polynomial(coeffs=coeffs[:1], batch_size=torch.Size([]))

        last_nonzero = torch.where(mask)[0][-1]
        return Polynomial(coeffs=coeffs[:last_nonzero + 1], batch_size=torch.Size([]))

    # For batched case, we need to keep same shape across batch
    # So just trim based on any batch element having non-zero
    mask = torch.abs(coeffs) > tol
    # Any non-zero across batch dimensions
    mask_any = mask.any(dim=tuple(range(coeffs.dim() - 1)))

    if not mask_any.any():
        return Polynomial(coeffs=coeffs[..., :1], batch_size=coeffs.shape[:-1])

    last_nonzero = torch.where(mask_any)[0][-1]
    return Polynomial(coeffs=coeffs[..., :last_nonzero + 1], batch_size=coeffs.shape[:-1])


def polynomial_equal(p: Polynomial, q: Polynomial, tol: float = 1e-8) -> Tensor:
    """Check polynomial equality within tolerance.

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to compare.
    tol : float
        Absolute tolerance for coefficient comparison.

    Returns
    -------
    Tensor
        Boolean tensor, shape matches broadcast of batch dims.
    """
    p_coeffs = p.coeffs
    q_coeffs = q.coeffs

    # Pad to same length
    p_deg = p_coeffs.shape[-1]
    q_deg = q_coeffs.shape[-1]
    max_deg = max(p_deg, q_deg)

    if p_deg < max_deg:
        pad_shape = list(p_coeffs.shape)
        pad_shape[-1] = max_deg - p_deg
        p_coeffs = torch.cat([p_coeffs, torch.zeros(pad_shape, dtype=p_coeffs.dtype, device=p_coeffs.device)], dim=-1)

    if q_deg < max_deg:
        pad_shape = list(q_coeffs.shape)
        pad_shape[-1] = max_deg - q_deg
        q_coeffs = torch.cat([q_coeffs, torch.zeros(pad_shape, dtype=q_coeffs.dtype, device=q_coeffs.device)], dim=-1)

    # Check all coefficients are within tolerance
    diff = torch.abs(p_coeffs - q_coeffs)
    return (diff <= tol).all(dim=-1)
```

**Step 2: Update __init__.py exports**

Add `polynomial_trim` and `polynomial_equal` to imports and `__all__`.

**Step 3: Run tests to verify they pass**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestPolynomialTrim tests/torchscience/polynomial/test__polynomial.py::TestPolynomialEqual -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): implement polynomial_trim and polynomial_equal"
```

---

## Phase 5: Final Integration and NumPy Comparison Tests

### Task 5.1: Add NumPy comparison tests

**Files:**
- Modify: `tests/torchscience/polynomial/test__polynomial.py`

**Step 1: Add NumPy comparison tests**

```python
# Add to tests/torchscience/polynomial/test__polynomial.py

class TestNumpyComparison:
    """Compare results against numpy.polynomial.Polynomial."""

    def test_add_matches_numpy(self):
        """Test polynomial addition matches NumPy."""
        import numpy as np
        from numpy.polynomial import Polynomial as NPoly

        from torchscience.polynomial import polynomial, polynomial_add

        c1 = [1.0, 2.0, 3.0]
        c2 = [4.0, 5.0]

        np_result = (NPoly(c1) + NPoly(c2)).coef
        torch_result = polynomial_add(
            polynomial(torch.tensor(c1)),
            polynomial(torch.tensor(c2))
        ).coeffs.numpy()

        np.testing.assert_allclose(torch_result, np_result)

    def test_multiply_matches_numpy(self):
        """Test polynomial multiplication matches NumPy."""
        import numpy as np
        from numpy.polynomial import Polynomial as NPoly

        from torchscience.polynomial import polynomial, polynomial_multiply

        c1 = [1.0, 2.0]
        c2 = [3.0, 4.0]

        np_result = (NPoly(c1) * NPoly(c2)).coef
        torch_result = polynomial_multiply(
            polynomial(torch.tensor(c1)),
            polynomial(torch.tensor(c2))
        ).coeffs.numpy()

        np.testing.assert_allclose(torch_result, np_result)

    def test_derivative_matches_numpy(self):
        """Test polynomial derivative matches NumPy."""
        import numpy as np
        from numpy.polynomial import Polynomial as NPoly

        from torchscience.polynomial import polynomial, polynomial_derivative

        c = [1.0, 2.0, 3.0, 4.0]

        np_result = NPoly(c).deriv().coef
        torch_result = polynomial_derivative(polynomial(torch.tensor(c))).coeffs.numpy()

        np.testing.assert_allclose(torch_result, np_result)

    def test_integral_matches_numpy(self):
        """Test polynomial integral matches NumPy."""
        import numpy as np
        from numpy.polynomial import Polynomial as NPoly

        from torchscience.polynomial import polynomial, polynomial_integral

        c = [1.0, 2.0, 3.0]
        a, b = 0.0, 2.0

        np_poly = NPoly(c)
        np_result = np_poly.integ()(b) - np_poly.integ()(a)
        torch_result = polynomial_integral(
            polynomial(torch.tensor(c)),
            torch.tensor(a),
            torch.tensor(b)
        ).item()

        np.testing.assert_allclose(torch_result, np_result)

    def test_roots_matches_numpy(self):
        """Test polynomial roots match NumPy."""
        import numpy as np
        from numpy.polynomial import Polynomial as NPoly

        from torchscience.polynomial import polynomial, polynomial_roots

        c = [2.0, -3.0, 1.0]  # (x-1)(x-2)

        np_roots = sorted(NPoly(c).roots())
        torch_roots = sorted(polynomial_roots(polynomial(torch.tensor(c, dtype=torch.float64))).real.numpy())

        np.testing.assert_allclose(torch_roots, np_roots, atol=1e-10)
```

**Step 2: Run NumPy comparison tests**

Run: `uv run python -m pytest tests/torchscience/polynomial/test__polynomial.py::TestNumpyComparison -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/torchscience/polynomial/test__polynomial.py
git commit -m "test(polynomial): add NumPy comparison tests"
```

---

### Task 5.2: Run full test suite and finalize

**Step 1: Run all polynomial tests**

Run: `uv run python -m pytest tests/torchscience/polynomial/ -v`
Expected: All tests PASS

**Step 2: Update final __init__.py with complete exports**

Verify `src/torchscience/polynomial/__init__.py` has all exports:

```python
"""Differentiable polynomial arithmetic for PyTorch tensors."""

from torchscience.polynomial._exceptions import (
    DegreeError,
    PolynomialError,
)
from torchscience.polynomial._polynomial import (
    Polynomial,
    polynomial,
    polynomial_add,
    polynomial_antiderivative,
    polynomial_degree,
    polynomial_derivative,
    polynomial_equal,
    polynomial_evaluate,
    polynomial_integral,
    polynomial_multiply,
    polynomial_negate,
    polynomial_scale,
    polynomial_subtract,
    polynomial_trim,
)
from torchscience.polynomial._roots import (
    polynomial_from_roots,
    polynomial_roots,
)

__all__ = [
    # Data types
    "Polynomial",
    # Constructors
    "polynomial",
    "polynomial_from_roots",
    # Arithmetic
    "polynomial_add",
    "polynomial_subtract",
    "polynomial_multiply",
    "polynomial_scale",
    "polynomial_negate",
    # Evaluation & calculus
    "polynomial_evaluate",
    "polynomial_derivative",
    "polynomial_antiderivative",
    "polynomial_integral",
    # Root finding
    "polynomial_roots",
    # Utilities
    "polynomial_degree",
    "polynomial_trim",
    "polynomial_equal",
    # Exceptions
    "PolynomialError",
    "DegreeError",
]
```

**Step 3: Final commit**

```bash
git add src/torchscience/polynomial/
git commit -m "feat(polynomial): complete polynomial module implementation"
```

**Step 4: Mark plan as complete**

Update the design document status if needed.
