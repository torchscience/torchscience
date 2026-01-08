# Orthogonal Polynomials Design

## Overview

Extend `torchscience.polynomial` with orthogonal polynomial classes following the established `ChebyshevT` pattern. Each family is an independent `@tensorclass` with full operation parity and independent implementations (no shared base classes or helpers).

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Rename strategy | Clean break | No deprecation aliases; simpler codebase |
| Domain handling | Retrofit to all | Add `DOMAIN` constant and validation to `ChebyshevPolynomialT` and all new classes |
| Code reuse | Copy-paste per class | Each class fully independent; ~29 files per class |
| Parameters | Tensor fields | Enables batched parameters with `torch.allclose` checks |
| Domain warnings | Evaluation-like ops | Warn on `evaluate`, `integral`, `weight` when outside domain |
| Unbounded linspace | Require explicit bounds | `linspace` raises error if `start`/`end` omitted for unbounded domains |
| Points function | Roots of P_n | Consistent across all families; standard Gaussian quadrature nodes |

## Scope

### Polynomial Families (10 new classes)

| Class | Parameters | Natural Domain | Notes |
|-------|------------|----------------|-------|
| `ChebyshevPolynomialU` | - | [-1, 1] | Second kind |
| `ChebyshevPolynomialV` | - | [-1, 1] | Third kind |
| `ChebyshevPolynomialW` | - | [-1, 1] | Fourth kind |
| `GegenbauerPolynomial` | λ > -1/2 | [-1, 1] | Ultraspherical |
| `JacobiPolynomial` | α, β > -1 | [-1, 1] | Most general |
| `LegendrePolynomial` | - | [-1, 1] | Spherical harmonics |
| `LaguerrePolynomial` | - | [0, ∞) | Quantum mechanics |
| `GeneralizedLaguerrePolynomial` | α > -1 | [0, ∞) | Associated Laguerre |
| `PhysicistsHermitePolynomialH` | - | (-∞, ∞) | H_n convention |
| `ProbabilistsHermitePolynomialHe` | - | (-∞, ∞) | He_n convention |

### Existing Classes (rename for consistency)

| Current Name | New Name |
|--------------|----------|
| `ChebyshevT` | `ChebyshevPolynomialT` |
| `_chebyshev_t/` | `_chebyshev_polynomial_t/` |

## Module Structure

```
src/torchscience/polynomial/
├── __init__.py
├── _exceptions.py                              # Shared exceptions
├── _polynomial/                                # Existing power basis
├── _chebyshev_polynomial_t/                    # Renamed from _chebyshev_t
├── _chebyshev_polynomial_u/
├── _chebyshev_polynomial_v/
├── _chebyshev_polynomial_w/
├── _gegenbauer_polynomial/
├── _jacobi_polynomial/
├── _legendre_polynomial/
├── _laguerre_polynomial/
├── _generalized_laguerre_polynomial/
├── _physicists_hermite_polynomial_h/           # H_n convention
└── _probabilists_hermite_polynomial_he/        # He_n convention
```

Each class directory follows one-file-per-operation pattern:

```
_legendre_polynomial/
├── __init__.py
├── _legendre_polynomial.py                  # Class definition
├── _legendre_polynomial_add.py
├── _legendre_polynomial_subtract.py
├── _legendre_polynomial_multiply.py
├── _legendre_polynomial_divmod.py
├── _legendre_polynomial_div.py
├── _legendre_polynomial_mod.py
├── _legendre_polynomial_scale.py
├── _legendre_polynomial_negate.py
├── _legendre_polynomial_pow.py
├── _legendre_polynomial_evaluate.py
├── _legendre_polynomial_derivative.py
├── _legendre_polynomial_antiderivative.py
├── _legendre_polynomial_integral.py
├── _legendre_polynomial_fit.py
├── _legendre_polynomial_interpolate.py
├── _legendre_polynomial_vandermonde.py
├── _legendre_polynomial_points.py
├── _legendre_polynomial_weight.py
├── _legendre_polynomial_roots.py
├── _legendre_polynomial_from_roots.py
├── _legendre_polynomial_companion.py
├── _legendre_polynomial_degree.py
├── _legendre_polynomial_trim.py
├── _legendre_polynomial_equal.py
├── _legendre_polynomial_mulx.py
├── _legendre_polynomial_linspace.py
├── _legendre_polynomial_to_polynomial.py
└── _legendre_polynomial_from_polynomial.py
```

## Class Definitions

### Non-parameterized Classes

```python
from tensordict.tensorclass import tensorclass
from torch import Tensor

@tensorclass
class LegendrePolynomial:
    """
    Legendre polynomial series.

    Represents f(x) = sum(c[k] * P_k(x)) where P_k are Legendre polynomials.

    The Legendre polynomials are orthogonal on [-1, 1] with weight w(x) = 1.

    Attributes:
        coeffs: Tensor of shape (...batch, N) where N = degree + 1.
                coeffs[..., k] is the coefficient for P_k(x).
    """
    coeffs: Tensor

    DOMAIN = (-1.0, 1.0)


@tensorclass
class PhysicistsHermitePolynomialH:
    """
    Physicists' Hermite polynomial series (H_n convention).

    Represents f(x) = sum(c[k] * H_k(x)) where H_k are physicists' Hermite polynomials.

    The physicists' Hermite polynomials are orthogonal on (-∞, ∞) with weight
    w(x) = exp(-x^2).

    Attributes:
        coeffs: Tensor of shape (...batch, N) where N = degree + 1.
                coeffs[..., k] is the coefficient for H_k(x).
    """
    coeffs: Tensor

    DOMAIN = (float('-inf'), float('inf'))


@tensorclass
class ProbabilistsHermitePolynomialHe:
    """
    Probabilists' Hermite polynomial series (He_n convention).

    Represents f(x) = sum(c[k] * He_k(x)) where He_k are probabilists' Hermite polynomials.

    The probabilists' Hermite polynomials are orthogonal on (-∞, ∞) with weight
    w(x) = exp(-x^2/2).

    Attributes:
        coeffs: Tensor of shape (...batch, N) where N = degree + 1.
                coeffs[..., k] is the coefficient for He_k(x).
    """
    coeffs: Tensor

    DOMAIN = (float('-inf'), float('inf'))


@tensorclass
class LaguerrePolynomial:
    """
    Laguerre polynomial series.

    Represents f(x) = sum(c[k] * L_k(x)) where L_k are Laguerre polynomials.

    The Laguerre polynomials are orthogonal on [0, ∞) with weight w(x) = exp(-x).

    Attributes:
        coeffs: Tensor of shape (...batch, N) where N = degree + 1.
                coeffs[..., k] is the coefficient for L_k(x).
    """
    coeffs: Tensor

    DOMAIN = (0.0, float('inf'))
```

### Parameterized Classes

```python
@tensorclass
class JacobiPolynomial:
    """
    Jacobi polynomial series.

    Represents f(x) = sum(c[k] * P_k^{(α,β)}(x)) where P_k^{(α,β)} are
    Jacobi polynomials with parameters α and β.

    The Jacobi polynomials are orthogonal on [-1, 1] with weight
    w(x) = (1-x)^α * (1+x)^β.

    Attributes:
        coeffs: Tensor of shape (...batch, N) where N = degree + 1.
        alpha: Parameter α, must be > -1. Tensor for batch support.
        beta: Parameter β, must be > -1. Tensor for batch support.
    """
    coeffs: Tensor
    alpha: Tensor
    beta: Tensor

    DOMAIN = (-1.0, 1.0)

    def __post_init__(self):
        if (self.alpha <= -1).any():
            raise ParameterError(f"alpha must be > -1, got {self.alpha}")
        if (self.beta <= -1).any():
            raise ParameterError(f"beta must be > -1, got {self.beta}")


@tensorclass
class GegenbauerPolynomial:
    """
    Gegenbauer (ultraspherical) polynomial series.

    Represents f(x) = sum(c[k] * C_k^{λ}(x)) where C_k^{λ} are
    Gegenbauer polynomials with parameter λ.

    The Gegenbauer polynomials are orthogonal on [-1, 1] with weight
    w(x) = (1-x^2)^{λ-1/2}.

    Attributes:
        coeffs: Tensor of shape (...batch, N) where N = degree + 1.
        lambda_: Parameter λ, must be > -1/2. Tensor for batch support.
    """
    coeffs: Tensor
    lambda_: Tensor

    DOMAIN = (-1.0, 1.0)

    def __post_init__(self):
        if (self.lambda_ <= -0.5).any():
            raise ParameterError(f"lambda must be > -1/2, got {self.lambda_}")


@tensorclass
class GeneralizedLaguerrePolynomial:
    """
    Generalized Laguerre polynomial series.

    Represents f(x) = sum(c[k] * L_k^{α}(x)) where L_k^{α} are
    generalized Laguerre polynomials with parameter α.

    The generalized Laguerre polynomials are orthogonal on [0, ∞) with
    weight w(x) = x^α * exp(-x).

    Attributes:
        coeffs: Tensor of shape (...batch, N) where N = degree + 1.
        alpha: Parameter α, must be > -1. Tensor for batch support.
    """
    coeffs: Tensor
    alpha: Tensor

    DOMAIN = (0.0, float('inf'))

    def __post_init__(self):
        if (self.alpha <= -1).any():
            raise ParameterError(f"alpha must be > -1, got {self.alpha}")
```

## Operation Set

Each class implements 29 operations with full parity:

### Arithmetic (9 operations)

| Operation | Description |
|-----------|-------------|
| `add` | Add two polynomials (pad shorter, sum coefficients) |
| `subtract` | Subtract two polynomials |
| `multiply` | Multiply two polynomials (convolution in basis) |
| `divmod` | Polynomial long division (quotient, remainder) |
| `div` | Quotient only |
| `mod` | Remainder only |
| `scale` | Scalar multiplication |
| `negate` | Negate all coefficients |
| `pow` | Integer exponentiation |

### Evaluation (1 operation)

| Operation | Description |
|-----------|-------------|
| `evaluate` | Evaluate at points using Clenshaw algorithm |

### Calculus (3 operations)

| Operation | Description |
|-----------|-------------|
| `derivative` | Compute derivative series |
| `antiderivative` | Compute antiderivative series |
| `integral` | Definite integral over interval |

### Fitting & Interpolation (4 operations)

| Operation | Description |
|-----------|-------------|
| `fit` | Least squares fit to data points |
| `interpolate` | Interpolate at specific points |
| `vandermonde` | Construct Vandermonde matrix |
| `points` | Return roots of P_n (Gaussian quadrature nodes) |

### Weight & Domain (2 operations)

| Operation | Description |
|-----------|-------------|
| `weight` | Evaluate weight function w(x) |
| `linspace` | Sample points in domain (requires explicit bounds for unbounded) |

### Roots (3 operations)

| Operation | Description |
|-----------|-------------|
| `roots` | Find roots via companion matrix eigenvalues |
| `from_roots` | Construct polynomial from roots |
| `companion` | Return companion matrix |

### Utilities (5 operations)

| Operation | Description |
|-----------|-------------|
| `degree` | Return polynomial degree |
| `trim` | Remove trailing near-zero coefficients |
| `equal` | Equality check with tolerance |
| `mulx` | Multiply by x (shift in basis) |
| `domain` | Return natural domain (class property) |

### Conversion (2 operations)

| Operation | Description |
|-----------|-------------|
| `to_polynomial` | Convert to power basis `Polynomial` |
| `from_polynomial` | Convert from power basis `Polynomial` |

## Evaluation Algorithms

Each family uses Clenshaw's algorithm with family-specific three-term recurrence:

```
P_n(x) = (A_n * x + B_n) * P_{n-1}(x) - C_n * P_{n-2}(x)
```

### Recurrence Coefficients

| Family | A_n | B_n | C_n |
|--------|-----|-----|-----|
| Legendre | (2n-1)/n | 0 | (n-1)/n |
| Chebyshev T | 2 (n>1), 1 (n=1) | 0 | 1 |
| Chebyshev U | 2 | 0 | 1 |
| Chebyshev V | 2 | 0 | 1 |
| Chebyshev W | 2 | 0 | 1 |
| Physicists Hermite H | 2 | 0 | 2(n-1) |
| Probabilists Hermite He | 1 | 0 | n-1 |
| Laguerre | -1/n | (2n-1)/n | (n-1)/n |
| Gen. Laguerre(α) | -1/n | (2n+α-1)/n | (n+α-1)/n |
| Gegenbauer(λ) | 2(n+λ-1)/n | 0 | (n+2λ-2)/n |
| Jacobi(α,β) | (see formula below) | (see formula below) | (see formula below) |

### Jacobi Recurrence

For Jacobi polynomials P_n^{(α,β)}(x):

```
a_n = 2(n+1)(n+α+β+1)(2n+α+β)
b_n = (2n+α+β+1)(α²-β²)
c_n = (2n+α+β)(2n+α+β+1)(2n+α+β+2)
d_n = 2(n+α)(n+β)(2n+α+β+2)

P_{n+1}(x) = ((b_n + c_n*x) * P_n(x) - d_n * P_{n-1}(x)) / a_n
```

### Clenshaw Algorithm Template

```python
def _clenshaw_legendre(coeffs: Tensor, x: Tensor) -> Tensor:
    """Evaluate Legendre series using Clenshaw's algorithm."""
    # coeffs: (...batch, N), x: (...points)
    # Returns: (...batch, ...points)

    n = coeffs.shape[-1]
    if n == 0:
        return torch.zeros_like(x)
    if n == 1:
        return coeffs[..., 0:1] * torch.ones_like(x)

    # Backward recurrence
    b_k = torch.zeros_like(x)      # b_{n+1} = 0
    b_k1 = torch.zeros_like(x)     # b_{n+2} = 0

    for k in range(n - 1, 0, -1):
        # Legendre: A_k = (2k-1)/k, B_k = 0, C_k = (k-1)/k
        A_k = (2 * k - 1) / k
        C_k = (k - 1) / k
        b_k, b_k1 = A_k * x * b_k - C_k * b_k1 + coeffs[..., k], b_k

    # Final step: P_0(x) = 1, P_1(x) = x
    return coeffs[..., 0] + x * b_k - 0.5 * b_k1
```

## Domain Handling

### Class-Level Domain Constants

```python
class LegendrePolynomial:
    DOMAIN = (-1.0, 1.0)

class LaguerrePolynomial:
    DOMAIN = (0.0, float('inf'))

class PhysicistsHermitePolynomialH:
    DOMAIN = (float('-inf'), float('inf'))
```

### Domain Behavior by Operation

| Operation | Bounded Domain | Unbounded Domain |
|-----------|----------------|------------------|
| `evaluate` | Warn if outside | No warning (always valid) |
| `integral` | Warn if bounds outside | No warning |
| `weight` | Warn if outside | Warn if outside valid range |
| `fit` | Error if points outside | Error if points outside |
| `linspace` | Uses DOMAIN bounds | **Requires explicit start/end** |

### Evaluation: Warn on Extrapolation

```python
import warnings

def legendre_polynomial_evaluate(
    p: LegendrePolynomial,
    x: Tensor
) -> Tensor:
    """Evaluate Legendre polynomial series at points x."""
    domain = LegendrePolynomial.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        warnings.warn(
            f"Evaluating LegendrePolynomial outside natural domain "
            f"[{domain[0]}, {domain[1]}]. Results may be numerically unstable.",
            stacklevel=2
        )

    return _clenshaw_legendre(p.coeffs, x)
```

### Fitting: Error on Out-of-Domain

```python
def legendre_polynomial_fit(
    x: Tensor,
    y: Tensor,
    degree: int
) -> LegendrePolynomial:
    """Fit Legendre polynomial series to data."""
    domain = LegendrePolynomial.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        raise DomainError(
            f"Fitting points must be in [{domain[0]}, {domain[1]}] "
            f"for LegendrePolynomial"
        )

    # Least squares fitting...
    V = legendre_polynomial_vandermonde(x, degree)
    coeffs, *_ = torch.linalg.lstsq(V, y)
    return LegendrePolynomial(coeffs)
```

### Linspace: Require Explicit Bounds for Unbounded Domains

```python
def physicists_hermite_polynomial_h_linspace(
    n: int,
    start: float | None = None,
    end: float | None = None,
    **kwargs
) -> Tensor:
    """Sample points for Hermite polynomial evaluation.

    For unbounded domains, start and end must be explicitly provided.
    """
    domain = PhysicistsHermitePolynomialH.DOMAIN

    if math.isinf(domain[0]) or math.isinf(domain[1]):
        if start is None or end is None:
            raise DomainError(
                f"PhysicistsHermitePolynomialH has unbounded domain {domain}. "
                f"Must provide explicit start and end arguments."
            )

    start = start if start is not None else domain[0]
    end = end if end is not None else domain[1]

    return torch.linspace(start, end, n, **kwargs)
```

## Error Handling

### Exception Hierarchy

```python
# src/torchscience/polynomial/_exceptions.py

class PolynomialError(Exception):
    """Base exception for polynomial operations."""
    pass


class DegreeError(PolynomialError):
    """Invalid polynomial degree."""
    pass


class DomainError(PolynomialError):
    """Operation outside valid domain."""
    pass


class ParameterError(PolynomialError):
    """Invalid polynomial parameters (α, β, λ)."""
    pass


class ParameterMismatchError(PolynomialError):
    """Arithmetic between polynomials with different parameters."""
    pass
```

### Parameter Validation

```python
@tensorclass
class JacobiPolynomial:
    coeffs: Tensor
    alpha: Tensor
    beta: Tensor

    def __post_init__(self):
        if (self.alpha <= -1).any():
            raise ParameterError(f"alpha must be > -1, got {self.alpha}")
        if (self.beta <= -1).any():
            raise ParameterError(f"beta must be > -1, got {self.beta}")
```

### Parameter Matching in Arithmetic

```python
def jacobi_polynomial_add(
    p1: JacobiPolynomial,
    p2: JacobiPolynomial
) -> JacobiPolynomial:
    """Add two Jacobi polynomial series."""
    if not torch.allclose(p1.alpha, p2.alpha) or not torch.allclose(p1.beta, p2.beta):
        raise ParameterMismatchError(
            f"Cannot add JacobiPolynomial with alpha={p1.alpha}, beta={p1.beta} "
            f"to JacobiPolynomial with alpha={p2.alpha}, beta={p2.beta}"
        )

    # Pad and add coefficients...
    n1, n2 = p1.coeffs.shape[-1], p2.coeffs.shape[-1]
    n = max(n1, n2)
    c1 = torch.nn.functional.pad(p1.coeffs, (0, n - n1))
    c2 = torch.nn.functional.pad(p2.coeffs, (0, n - n2))

    return JacobiPolynomial(c1 + c2, p1.alpha, p1.beta)
```

## Conversions

### Hub-and-Spoke via Power Basis

All conversions go through the power basis `Polynomial` class:

```
LegendrePolynomial ─────────┐
ChebyshevPolynomialU ───────┤
JacobiPolynomial ───────────┼──── Polynomial (hub) ────┼─── LegendrePolynomial
GegenbauerPolynomial ───────┤                          ├─── ChebyshevPolynomialU
PhysicistsHermitePolynomialH┘                          └─── etc.
```

To convert between any two families:
```python
# Legendre → Hermite
legendre_poly = LegendrePolynomial(coeffs)
power_poly = legendre_polynomial_to_polynomial(legendre_poly)
hermite_poly = physicists_hermite_polynomial_h_from_polynomial(power_poly)
```

### Conversion Functions

Each class provides two conversion functions:

```python
def legendre_polynomial_to_polynomial(p: LegendrePolynomial) -> Polynomial:
    """Convert Legendre series to power basis polynomial."""
    # Use recurrence to build power basis representation
    ...

def legendre_polynomial_from_polynomial(p: Polynomial) -> LegendrePolynomial:
    """Convert power basis polynomial to Legendre series."""
    # Project onto Legendre basis using inner products
    ...
```

### Conversion Formulas

| From | To Power Basis | Method |
|------|---------------|--------|
| Legendre | Σ c_k P_k(x) | Recurrence expansion |
| Chebyshev T | Σ c_k T_k(x) | DCT-based or recurrence |
| Chebyshev U | Σ c_k U_k(x) | Recurrence expansion |
| Physicists Hermite H | Σ c_k H_k(x) | Recurrence expansion |
| Probabilists Hermite He | Σ c_k He_k(x) | Recurrence expansion |
| Laguerre | Σ c_k L_k(x) | Recurrence expansion |
| Jacobi | Σ c_k P_k^{(α,β)}(x) | Recurrence expansion |
| Gegenbauer | Σ c_k C_k^{λ}(x) | Recurrence expansion |

## Testing Strategy

### Test File Structure

```
tests/torchscience/polynomial/
├── test__chebyshev_polynomial_t.py               # Renamed
├── test__chebyshev_polynomial_u.py
├── test__chebyshev_polynomial_v.py
├── test__chebyshev_polynomial_w.py
├── test__legendre_polynomial.py
├── test__gegenbauer_polynomial.py
├── test__jacobi_polynomial.py
├── test__laguerre_polynomial.py
├── test__generalized_laguerre_polynomial.py
├── test__physicists_hermite_polynomial_h.py
└── test__probabilists_hermite_polynomial_he.py
```

### Test Categories

| Category | What it tests | Method |
|----------|---------------|--------|
| Forward correctness | Evaluation matches scipy | `scipy.special.eval_legendre`, etc. |
| Gradient (1st order) | Autograd works | `torch.autograd.gradcheck` |
| Gradient (2nd order) | Hessians work | `torch.autograd.gradgradcheck` |
| Special values | Endpoints, zeros | `P_n(1) = 1` for Legendre, etc. |
| Domain warnings | Out-of-domain warns | `pytest.warns(UserWarning)` |
| Domain errors | Fit outside domain | `pytest.raises(DomainError)` |
| Parameter validation | Invalid α, β, λ | `pytest.raises(ParameterError)` |
| Parameter mismatch | Arithmetic with different params | `pytest.raises(ParameterMismatchError)` |
| Conversion roundtrip | to_polynomial → from_polynomial | `torch.testing.assert_close` |
| Arithmetic identities | p + (-p) = 0, p * 1 = p | Standard algebraic laws |

### Example Tests

```python
import scipy.special
import torch
import pytest

from torchscience.polynomial import (
    LegendrePolynomial,
    legendre_polynomial_evaluate,
    legendre_polynomial_fit,
    JacobiPolynomial,
    jacobi_polynomial_add,
)
from torchscience.polynomial._exceptions import (
    DomainError,
    ParameterError,
    ParameterMismatchError,
)


def test_legendre_evaluate():
    coeffs = torch.tensor([1.0, 2.0, 3.0])  # 1*P_0 + 2*P_1 + 3*P_2
    p = LegendrePolynomial(coeffs)
    x = torch.linspace(-1, 1, 100)

    actual = legendre_polynomial_evaluate(p, x)

    # Reference using scipy
    expected = sum(
        c * scipy.special.eval_legendre(k, x.numpy())
        for k, c in enumerate(coeffs.numpy())
    )

    torch.testing.assert_close(actual, torch.from_numpy(expected))


def test_legendre_evaluate_gradient():
    coeffs = torch.randn(5, dtype=torch.float64, requires_grad=True)
    x = torch.linspace(-0.9, 0.9, 10, dtype=torch.float64)

    def func(c):
        return legendre_polynomial_evaluate(LegendrePolynomial(c), x)

    torch.autograd.gradcheck(func, coeffs, raise_exception=True)


def test_legendre_evaluate_hessian():
    coeffs = torch.randn(5, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([0.5], dtype=torch.float64)

    def func(c):
        return legendre_polynomial_evaluate(LegendrePolynomial(c), x)

    torch.autograd.gradgradcheck(func, coeffs, raise_exception=True)


def test_legendre_at_endpoints():
    # P_n(1) = 1 for all n
    coeffs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    p = LegendrePolynomial(coeffs)

    result = legendre_polynomial_evaluate(p, torch.tensor([1.0]))
    expected = torch.tensor([5.0])  # sum of coefficients

    torch.testing.assert_close(result, expected)


def test_legendre_evaluate_warns_outside_domain():
    p = LegendrePolynomial(torch.tensor([1.0, 2.0]))
    x = torch.tensor([2.0])  # Outside [-1, 1]

    with pytest.warns(UserWarning, match="outside natural domain"):
        legendre_polynomial_evaluate(p, x)


def test_legendre_fit_errors_outside_domain():
    x = torch.tensor([2.0, 3.0])  # Outside [-1, 1]
    y = torch.tensor([1.0, 2.0])

    with pytest.raises(DomainError):
        legendre_polynomial_fit(x, y, degree=3)


def test_jacobi_invalid_alpha():
    coeffs = torch.tensor([1.0, 2.0])

    with pytest.raises(ParameterError, match="alpha must be > -1"):
        JacobiPolynomial(coeffs, alpha=torch.tensor(-2.0), beta=torch.tensor(0.0))


def test_jacobi_parameter_mismatch():
    p1 = JacobiPolynomial(torch.tensor([1.0]), alpha=torch.tensor(0.5), beta=torch.tensor(0.5))
    p2 = JacobiPolynomial(torch.tensor([1.0]), alpha=torch.tensor(1.0), beta=torch.tensor(1.0))

    with pytest.raises(ParameterMismatchError):
        jacobi_polynomial_add(p1, p2)


def test_legendre_conversion_roundtrip():
    original = LegendrePolynomial(torch.tensor([1.0, 2.0, 3.0]))

    power = legendre_polynomial_to_polynomial(original)
    recovered = legendre_polynomial_from_polynomial(power)

    torch.testing.assert_close(original.coeffs, recovered.coeffs)
```

## Implementation Order

### Phase 1: Foundation

1. **Rename existing ChebyshevT** → `ChebyshevPolynomialT`
   - Rename directory `_chebyshev_t/` → `_chebyshev_polynomial_t/`
   - Rename class and all operation files
   - Add `DOMAIN = (-1.0, 1.0)` constant
   - Add domain validation to `evaluate`, `integral`, `weight`, `fit`
   - Update imports and tests

2. **Implement LegendrePolynomial**
   - Simplest classical polynomial (no parameters)
   - Validates the full operation pattern
   - 29 operation files

### Phase 2: Chebyshev Family

3. **ChebyshevPolynomialU** - Second kind
4. **ChebyshevPolynomialV** - Third kind
5. **ChebyshevPolynomialW** - Fourth kind

All very similar to ChebyshevPolynomialT with minor recurrence differences.

### Phase 3: Parameterized Bounded Domain

6. **GegenbauerPolynomial** - Single parameter λ
7. **JacobiPolynomial** - Two parameters α, β (most general)

Validates parameter handling and mismatch errors.

### Phase 4: Unbounded Domains

8. **PhysicistsHermitePolynomialH** - H_n convention, domain (-∞, ∞)
9. **ProbabilistsHermitePolynomialHe** - He_n convention
10. **LaguerrePolynomial** - Domain [0, ∞)
11. **GeneralizedLaguerrePolynomial** - With α parameter

Validates unbounded domain handling and explicit bounds requirement for `linspace`.

## Summary

| Metric | Count |
|--------|-------|
| New polynomial classes | 10 |
| Existing class renames | 1 (ChebyshevT → ChebyshevPolynomialT) |
| Operations per class | 29 |
| New operation functions | ~290 |
| Test files | ~11 |
| Conversion functions | 22 (11 classes × 2 directions) |
