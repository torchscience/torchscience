"""Core polynomial operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._exceptions import PolynomialError

if TYPE_CHECKING:
    pass


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

    Operator overloading:
        p + q    # polynomial_add(p, q)
        p - q    # polynomial_subtract(p, q)
        p * q    # polynomial_multiply(p, q)
        -p       # polynomial_negate(p)
        p(x)     # polynomial_evaluate(p, x)
    """

    coeffs: Tensor

    def __add__(self, other: "Polynomial") -> "Polynomial":
        return polynomial_add(self, other)

    def __radd__(self, other: "Polynomial") -> "Polynomial":
        return polynomial_add(other, self)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        return polynomial_subtract(self, other)

    def __rsub__(self, other: "Polynomial") -> "Polynomial":
        return polynomial_subtract(other, self)

    def __mul__(self, other: Union["Polynomial", Tensor]) -> "Polynomial":
        if isinstance(other, Polynomial):
            return polynomial_multiply(self, other)
        return polynomial_scale(self, other)

    def __rmul__(self, other: Union["Polynomial", Tensor]) -> "Polynomial":
        if isinstance(other, Polynomial):
            return polynomial_multiply(other, self)
        return polynomial_scale(self, other)

    def __neg__(self) -> "Polynomial":
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

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> p.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError("Polynomial must have at least one coefficient")
    return Polynomial(coeffs=coeffs)


# ============================================================================
# Arithmetic Operations
# ============================================================================


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
    n_p = p_coeffs.shape[-1]
    n_q = q_coeffs.shape[-1]

    if n_p < n_q:
        padding = [0] * (2 * (p_coeffs.dim() - 1)) + [0, n_q - n_p]
        p_coeffs = torch.nn.functional.pad(p_coeffs, padding)
    elif n_q < n_p:
        padding = [0] * (2 * (q_coeffs.dim() - 1)) + [0, n_p - n_q]
        q_coeffs = torch.nn.functional.pad(q_coeffs, padding)

    return Polynomial(coeffs=p_coeffs + q_coeffs)


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
    p_coeffs = p.coeffs
    q_coeffs = q.coeffs

    # Pad shorter polynomial with zeros
    n_p = p_coeffs.shape[-1]
    n_q = q_coeffs.shape[-1]

    if n_p < n_q:
        padding = [0] * (2 * (p_coeffs.dim() - 1)) + [0, n_q - n_p]
        p_coeffs = torch.nn.functional.pad(p_coeffs, padding)
    elif n_q < n_p:
        padding = [0] * (2 * (q_coeffs.dim() - 1)) + [0, n_p - n_q]
        q_coeffs = torch.nn.functional.pad(q_coeffs, padding)

    return Polynomial(coeffs=p_coeffs - q_coeffs)


def polynomial_negate(p: Polynomial) -> Polynomial:
    """Negate polynomial.

    Returns
    -------
    Polynomial
        Negated polynomial -p.
    """
    return Polynomial(coeffs=-p.coeffs)


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
    # Ensure c can broadcast with coeffs
    if c.dim() == 0:
        return Polynomial(coeffs=p.coeffs * c)
    else:
        # c broadcasts with batch dimensions, not coefficient dimension
        return Polynomial(coeffs=p.coeffs * c.unsqueeze(-1))


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

    # Handle batch dimensions
    # We need to broadcast batch dimensions and convolve coefficient dimension

    # Get shapes
    p_batch = p_coeffs.shape[:-1]
    q_batch = q_coeffs.shape[:-1]
    n_p = p_coeffs.shape[-1]
    n_q = q_coeffs.shape[-1]

    # Broadcast batch dimensions
    broadcast_batch = torch.broadcast_shapes(p_batch, q_batch)

    # Expand to broadcast shape
    p_expanded = p_coeffs.expand(*broadcast_batch, n_p)
    q_expanded = q_coeffs.expand(*broadcast_batch, n_q)

    # For non-batched case, use explicit convolution
    if len(broadcast_batch) == 0:
        # Simple 1D convolution
        n_out = n_p + n_q - 1
        result = torch.zeros(
            n_out, dtype=p_coeffs.dtype, device=p_coeffs.device
        )
        for i in range(n_p):
            for j in range(n_q):
                result[i + j] = result[i + j] + p_expanded[i] * q_expanded[j]
        return Polynomial(coeffs=result)

    # For batched case, use explicit convolution per batch element
    # Flatten batch dimensions
    batch_size = broadcast_batch.numel() if len(broadcast_batch) > 0 else 1
    p_flat = p_expanded.reshape(batch_size, n_p)
    q_flat = q_expanded.reshape(batch_size, n_q)

    n_out = n_p + n_q - 1
    result = torch.zeros(
        batch_size, n_out, dtype=p_coeffs.dtype, device=p_coeffs.device
    )

    for i in range(n_p):
        for j in range(n_q):
            result[:, i + j] = result[:, i + j] + p_flat[:, i] * q_flat[:, j]

    # Reshape back to broadcast batch dimensions
    result = result.reshape(*broadcast_batch, n_out)
    return Polynomial(coeffs=result)


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
        Returns number of coefficients minus 1.

    Notes
    -----
    This returns the formal degree (len(coeffs) - 1), not the actual degree
    which would require checking for trailing zeros. Use polynomial_trim
    first if you need the actual degree.
    """
    return torch.tensor(p.coeffs.shape[-1] - 1, device=p.coeffs.device)


# ============================================================================
# Evaluation and Calculus
# ============================================================================


def polynomial_evaluate(p: Polynomial, x: Tensor) -> Tensor:
    """Evaluate polynomial at points using Horner's method.

    Parameters
    ----------
    p : Polynomial
        Polynomial with coefficients shape (...batch, N).
    x : Tensor
        Evaluation points, shape (...x_batch). The result shape is
        (...batch, ...x_batch) where batch dimensions of p broadcast
        with x.

    Returns
    -------
    Tensor
        Values p(x), shape is broadcast of p's batch dims with x's shape.

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> polynomial_evaluate(p, torch.tensor([0.0, 1.0, 2.0]))
    tensor([ 1.,  6., 17.])
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    # For non-batched polynomial (1D coeffs), just do standard Horner
    if coeffs.dim() == 1:
        result = coeffs[n - 1].expand_as(x).clone()
        for i in range(n - 2, -1, -1):
            result = result * x + coeffs[i]
        return result

    # For batched polynomial, we need to handle the broadcasting more carefully
    # coeffs shape: (...batch, N)
    # x shape: (...x_batch)
    # result shape: (...batch, ...x_batch)

    batch_shape = coeffs.shape[:-1]  # (...batch)
    x_shape = x.shape  # (...x_batch)

    # Reshape coeffs to (...batch, 1, 1, ..., N) to broadcast with x
    # Add x.dim() singleton dimensions before the coefficient dimension
    coeffs_expanded = coeffs
    for _ in range(x.dim()):
        # Insert dimension before the last (coefficient) dimension
        coeffs_expanded = coeffs_expanded.unsqueeze(-2)
    # Now coeffs_expanded has shape (...batch, 1, 1, ..., N)

    # Reshape x to (1, 1, ..., ...x_batch) to broadcast with batch dims
    x_expanded = x
    for _ in range(len(batch_shape)):
        x_expanded = x_expanded.unsqueeze(0)
    # Now x_expanded has shape (1, 1, ..., ...x_batch)

    # Start Horner's method
    result = coeffs_expanded[
        ..., n - 1
    ]  # shape: (...batch, 1, 1, ...) or broadcast result

    for i in range(n - 2, -1, -1):
        result = result * x_expanded + coeffs_expanded[..., i]

    return result


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

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> polynomial_derivative(p).coeffs  # 2 + 6x
    tensor([2., 6.])
    """
    coeffs = p.coeffs

    for _ in range(order):
        n = coeffs.shape[-1]
        if n <= 1:
            # Derivative of constant is zero
            return Polynomial(
                coeffs=torch.zeros(
                    *coeffs.shape[:-1],
                    1,
                    dtype=coeffs.dtype,
                    device=coeffs.device,
                )
            )

        # d/dx (a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n)
        # = a_1 + 2*a_2*x + 3*a_3*x^2 + ... + n*a_n*x^(n-1)
        # new_coeffs[i] = (i+1) * old_coeffs[i+1]
        indices = torch.arange(1, n, device=coeffs.device, dtype=coeffs.dtype)
        new_coeffs = coeffs[..., 1:] * indices

        coeffs = new_coeffs

    return Polynomial(coeffs=coeffs)


def polynomial_antiderivative(
    p: Polynomial, constant: Union[Tensor, float] = 0.0
) -> Polynomial:
    """Compute antiderivative (indefinite integral).

    Parameters
    ----------
    p : Polynomial
        Input polynomial.
    constant : Tensor or float
        Integration constant (default 0).

    Returns
    -------
    Polynomial
        Antiderivative with given constant term. Degree increases by 1.

    Examples
    --------
    >>> p = polynomial(torch.tensor([2.0, 6.0]))  # 2 + 6x
    >>> polynomial_antiderivative(p).coeffs  # 0 + 2x + 3x^2
    tensor([0., 2., 3.])
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    # Integral of (a_0 + a_1*x + ... + a_n*x^n)
    # = C + a_0*x + a_1*x^2/2 + a_2*x^3/3 + ... + a_n*x^(n+1)/(n+1)
    # new_coeffs[0] = constant
    # new_coeffs[i+1] = old_coeffs[i] / (i+1)

    indices = torch.arange(1, n + 1, device=coeffs.device, dtype=coeffs.dtype)
    integrated = coeffs / indices

    # Handle constant term
    if isinstance(constant, Tensor):
        c = constant
    else:
        c = torch.tensor(constant, dtype=coeffs.dtype, device=coeffs.device)

    # Expand constant to match batch dimensions
    if coeffs.dim() > 1 and c.dim() == 0:
        c = c.expand(*coeffs.shape[:-1])

    if c.dim() == 0:
        c = c.unsqueeze(-1)
    else:
        c = c.unsqueeze(-1)

    new_coeffs = torch.cat([c, integrated], dim=-1)
    return Polynomial(coeffs=new_coeffs)


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
        Definite integral integral_a^b p(x) dx.

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 0.0, 1.0]))  # 1 + x^2
    >>> polynomial_integral(p, torch.tensor(0.0), torch.tensor(1.0))
    tensor(1.3333)  # 1 + 1/3
    """
    # Compute antiderivative (with constant 0)
    anti = polynomial_antiderivative(p, 0.0)

    # Evaluate at bounds and subtract
    return polynomial_evaluate(anti, b) - polynomial_evaluate(anti, a)
