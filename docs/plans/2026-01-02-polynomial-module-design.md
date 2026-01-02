# torchscience.polynomial Module Design

**Goal:** Implement differentiable polynomial arithmetic for PyTorch tensors with full autograd support.

**Architecture:** Pure Python implementation using tensorclass data container. Ascending coefficient order (NumPy convention), batch dimensions first. All operations as pure functions for vmap/jacrev compatibility.

**Tech Stack:** PyTorch, tensordict (tensorclass), numpy (test comparisons only)

---

## Module Structure

```
src/torchscience/polynomial/
├── __init__.py
├── _exceptions.py          # PolynomialError, DegreeError
├── _polynomial.py          # Polynomial tensorclass + core functions
└── _roots.py               # poly_roots (companion matrix)

tests/torchscience/polynomial/
├── __init__.py
├── test__polynomial.py     # Core operations
└── test__roots.py          # Root finding
```

---

## Data Types

### Polynomial

```python
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

    Examples
    --------
    Single polynomial 1 + 2x + 3x^2:
        Polynomial(coeffs=torch.tensor([1.0, 2.0, 3.0]))

    Batch of 2 polynomials:
        Polynomial(coeffs=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        # First: 1 + 2x, Second: 3 + 4x
    """
    coeffs: Tensor
```

**Design decisions:**
- Ascending powers: `coeffs[..., i]` is coefficient of `x^i` (NumPy convention)
- Batch-first: Shape `(B1, B2, ..., N)` matches torchscience patterns
- tensorclass: Automatic batching, device/dtype handling, vmap compatibility

---

## Exceptions

```python
class PolynomialError(Exception):
    """Base exception for polynomial operations."""
    pass

class DegreeError(PolynomialError):
    """Raised when degree is invalid for operation."""
    pass
```

---

## Public API

### Constructors

```python
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

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> p.coeffs
    tensor([1., 2., 3.])
    """

def poly_from_roots(roots: Tensor) -> Polynomial:
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

    Examples
    --------
    >>> roots = torch.tensor([1.0, 2.0])  # (x-1)(x-2) = x^2 - 3x + 2
    >>> p = poly_from_roots(roots)
    >>> p.coeffs
    tensor([2., -3., 1.])
    """
```

### Arithmetic Operations

```python
def poly_add(p: Polynomial, q: Polynomial) -> Polynomial:
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

def poly_sub(p: Polynomial, q: Polynomial) -> Polynomial:
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

def poly_mul(p: Polynomial, q: Polynomial) -> Polynomial:
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

def poly_scale(p: Polynomial, c: Tensor) -> Polynomial:
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

def poly_neg(p: Polynomial) -> Polynomial:
    """Negate polynomial.

    Returns
    -------
    Polynomial
        Negated polynomial -p.
    """
```

**Implementation notes:**
- `poly_add`/`poly_sub`: Pad shorter coefficient tensor with zeros, then add/subtract
- `poly_mul`: Use explicit convolution loop or `torch.conv1d` for batched case
- All operations preserve dtype and device from inputs
- Autograd flows through naturally

### Evaluation and Calculus

```python
def poly_evaluate(p: Polynomial, x: Tensor) -> Tensor:
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

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> poly_evaluate(p, torch.tensor([0.0, 1.0, 2.0]))
    tensor([ 1.,  6., 17.])
    """

def poly_derivative(p: Polynomial, order: int = 1) -> Polynomial:
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

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> poly_derivative(p).coeffs  # 2 + 6x
    tensor([2., 6.])
    """

def poly_antiderivative(p: Polynomial, constant: Tensor = 0.0) -> Polynomial:
    """Compute antiderivative (indefinite integral).

    Parameters
    ----------
    p : Polynomial
        Input polynomial.
    constant : Tensor
        Integration constant (default 0).

    Returns
    -------
    Polynomial
        Antiderivative with given constant term. Degree increases by 1.

    Examples
    --------
    >>> p = polynomial(torch.tensor([2.0, 6.0]))  # 2 + 6x
    >>> poly_antiderivative(p).coeffs  # 0 + 2x + 3x^2
    tensor([0., 2., 3.])
    """

def poly_integral(p: Polynomial, a: Tensor, b: Tensor) -> Tensor:
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

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 0.0, 1.0]))  # 1 + x^2
    >>> poly_integral(p, torch.tensor(0.0), torch.tensor(1.0))
    tensor(1.3333)  # 1 + 1/3
    """
```

**Implementation:**
- Horner's method: `result = c[n]; for i in range(n-1, -1, -1): result = result * x + c[i]`
- Derivative: `new_coeffs[..., i] = (i+1) * coeffs[..., i+1]`
- Antiderivative: `new_coeffs[..., i+1] = coeffs[..., i] / (i+1)`
- Definite integral: Evaluate antiderivative at bounds, subtract

### Root Finding

```python
def poly_roots(p: Polynomial) -> Tensor:
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

    Examples
    --------
    >>> p = polynomial(torch.tensor([2.0, -3.0, 1.0]))  # (x-1)(x-2)
    >>> poly_roots(p)
    tensor([1.+0.j, 2.+0.j])

    Notes
    -----
    Uses companion matrix method:
    - Construct companion matrix from normalized coefficients
    - Compute eigenvalues via torch.linalg.eigvals
    - Supports autograd through eigenvalue computation

    For high-degree polynomials (>20), use float64 for accuracy.
    """
```

**Companion matrix construction:**
```
For p(x) = a_0 + a_1*x + ... + a_n*x^n:

C = [[0, 0, ..., 0, -a_0/a_n],
     [1, 0, ..., 0, -a_1/a_n],
     [0, 1, ..., 0, -a_2/a_n],
     [...                   ],
     [0, 0, ..., 1, -a_{n-1}/a_n]]

Eigenvalues of C are roots of p(x).
```

### Utilities

```python
def poly_degree(p: Polynomial) -> Tensor:
    """Return degree of polynomial(s).

    Parameters
    ----------
    p : Polynomial
        Input polynomial.

    Returns
    -------
    Tensor
        Degree, shape matches batch dimensions.
        Zero polynomial has degree 0 (or -1, TBD).
    """

def poly_trim(p: Polynomial, tol: float = 0.0) -> Polynomial:
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

def poly_equal(p: Polynomial, q: Polynomial, tol: float = 1e-8) -> Tensor:
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
```

---

## Error Handling

**Strict on construction:**
- `polynomial()` raises `PolynomialError` if coefficients tensor is empty
- `poly_roots()` raises `DegreeError` for constant/zero polynomials

**Natural results from operations:**
- `poly_derivative()` of constant returns `Polynomial([0.0])`
- `poly_add()` with different degrees pads with zeros
- Operations always return valid `Polynomial` instances

---

## Testing Strategy

**Correctness tests:**
- Compare against `numpy.polynomial.Polynomial` for all operations
- Property-based tests: `poly_evaluate(poly_derivative(p), x)` equals finite difference
- Round-trip: `poly_from_roots(poly_roots(p))` recovers `p` (up to scaling)

**Autograd tests:**
- `torch.autograd.gradcheck` for all differentiable operations
- `torch.autograd.gradgradcheck` for second-order derivatives
- Verify gradients flow through evaluation, arithmetic, and roots

**Edge cases:**
- Zero polynomial
- Constant polynomial
- High-degree polynomials (numerical stability)
- Complex coefficients and roots
- Batched operations with broadcasting

---

## Future Extensions (Not in Initial Scope)

- **Division:** `poly_divmod`, `poly_mod`, `poly_gcd`
- **Composition:** `poly_compose(p, q)` for `p(q(x))`
- **Power:** `poly_pow(p, n)` for `p(x)^n`
- **Orthogonal polynomials:** Legendre, Hermite, Laguerre, Jacobi families
- **Basis conversion:** `poly_from_chebyshev`, `poly_to_chebyshev`
- **Fitting:** `poly_fit` for least-squares polynomial fitting
- **Rational functions:** `RationalFunction` for p(x)/q(x)

---

## Implementation Phases

### Phase 1: Module Structure
- Create directory structure
- Implement exceptions
- Implement `Polynomial` tensorclass
- Implement `polynomial()` constructor

### Phase 2: Core Arithmetic
- `poly_add`, `poly_sub`, `poly_neg`
- `poly_scale`, `poly_mul`
- `poly_degree`

### Phase 3: Evaluation and Calculus
- `poly_evaluate` (Horner's method)
- `poly_derivative`, `poly_antiderivative`
- `poly_integral`

### Phase 4: Root Finding and Utilities
- `poly_roots` (companion matrix)
- `poly_from_roots`
- `poly_trim`, `poly_equal`

### Phase 5: Testing and Polish
- Comprehensive test suite
- NumPy comparison tests
- Autograd verification
- Documentation and examples