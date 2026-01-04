"""Polynomial division operations."""

import torch

from torchscience.polynomial._exceptions import DegreeError
from torchscience.polynomial._polynomial import Polynomial, polynomial


def polynomial_divmod(
    p: Polynomial, q: Polynomial
) -> tuple[Polynomial, Polynomial]:
    """Divide polynomial p by q, returning quotient and remainder.

    Computes quotient and remainder such that p = q * quotient + remainder,
    where deg(remainder) < deg(q).

    Parameters
    ----------
    p : Polynomial
        Dividend polynomial.
    q : Polynomial
        Divisor polynomial. Leading coefficient must be non-zero.

    Returns
    -------
    quotient : Polynomial
        Quotient of division.
    remainder : Polynomial
        Remainder of division.

    Raises
    ------
    DegreeError
        If divisor is zero polynomial.

    Examples
    --------
    >>> p = polynomial(torch.tensor([-1.0, 0.0, 0.0, 1.0]))  # x^3 - 1
    >>> q = polynomial(torch.tensor([-1.0, 1.0]))  # x - 1
    >>> quot, rem = polynomial_divmod(p, q)
    >>> quot.coeffs  # x^2 + x + 1
    tensor([1., 1., 1.])
    """
    p_coeffs = p.coeffs
    q_coeffs = q.coeffs

    # Get degrees
    deg_p = p_coeffs.shape[-1] - 1
    deg_q = q_coeffs.shape[-1] - 1

    # Check for zero divisor
    leading_q = q_coeffs[..., -1]
    if torch.all(leading_q == 0):
        raise DegreeError("Cannot divide by zero polynomial")

    # If dividend degree < divisor degree, quotient is 0, remainder is dividend
    if deg_p < deg_q:
        zero_shape = list(p_coeffs.shape)
        zero_shape[-1] = 1
        zero_coeffs = torch.zeros(
            zero_shape, dtype=p_coeffs.dtype, device=p_coeffs.device
        )
        return polynomial(zero_coeffs), p

    # Polynomial long division algorithm (autograd-compatible, no in-place ops)
    # We collect quotient coefficients in a list and build tensors without in-place ops
    quot_len = deg_p - deg_q + 1

    # Work with the remainder, starting from dividend
    remainder = p_coeffs

    # Collect quotient coefficients from highest to lowest degree
    quot_coeffs_list = []

    for i in range(quot_len - 1, -1, -1):
        # Current position in remainder (from the end)
        pos = deg_q + i

        # Quotient coefficient at position i
        q_coeff = remainder[..., pos] / leading_q
        quot_coeffs_list.append(q_coeff)

        # Create a tensor to subtract: q_coeff * q, shifted by i positions
        # Build the subtraction tensor without in-place operations
        # We need to subtract q_coeff * q * x^i from remainder

        # Create padded version of q * q_coeff
        # q has deg_q + 1 coefficients, we need to shift it by i positions
        n_rem = remainder.shape[-1]

        # Build the subtraction polynomial with proper shape
        # Left padding: i zeros, then q * q_coeff, then right padding to match remainder size
        left_pad = i
        right_pad = n_rem - (left_pad + deg_q + 1)

        # q_coeff has shape (...,), q_coeffs has shape (..., deg_q + 1)
        # We need q_coeff.unsqueeze(-1) * q_coeffs to get (..., deg_q + 1)
        scaled_q = q_coeff.unsqueeze(-1) * q_coeffs

        # Pad to match remainder shape
        padding = [left_pad, right_pad]
        subtraction = torch.nn.functional.pad(scaled_q, padding)

        # Update remainder (no in-place operation)
        remainder = remainder - subtraction

    # Reverse the quotient coefficients (we collected from high to low degree)
    quot_coeffs_list.reverse()

    # Stack quotient coefficients
    quot_coeffs = torch.stack(quot_coeffs_list, dim=-1)

    # Trim remainder to proper size (deg_q coefficients or 1 if all zero)
    remainder_coeffs = (
        remainder[..., :deg_q] if deg_q > 0 else remainder[..., :1]
    )

    # Ensure at least one coefficient
    if remainder_coeffs.shape[-1] == 0:
        rem_shape = list(p_coeffs.shape)
        rem_shape[-1] = 1
        remainder_coeffs = torch.zeros(
            rem_shape, dtype=p_coeffs.dtype, device=p_coeffs.device
        )

    return polynomial(quot_coeffs), polynomial(remainder_coeffs)


def polynomial_div(p: Polynomial, q: Polynomial) -> Polynomial:
    """Return quotient of polynomial division.

    Convenience wrapper around polynomial_divmod that returns only the quotient.

    Parameters
    ----------
    p : Polynomial
        Dividend polynomial.
    q : Polynomial
        Divisor polynomial.

    Returns
    -------
    Polynomial
        Quotient of p / q.

    Examples
    --------
    >>> p = polynomial(torch.tensor([-1.0, 0.0, 0.0, 1.0]))  # x^3 - 1
    >>> q = polynomial(torch.tensor([-1.0, 1.0]))  # x - 1
    >>> polynomial_div(p, q).coeffs
    tensor([1., 1., 1.])
    """
    quotient, _ = polynomial_divmod(p, q)
    return quotient


def polynomial_mod(p: Polynomial, q: Polynomial) -> Polynomial:
    """Return remainder of polynomial division.

    Convenience wrapper around polynomial_divmod that returns only the remainder.

    Parameters
    ----------
    p : Polynomial
        Dividend polynomial.
    q : Polynomial
        Divisor polynomial.

    Returns
    -------
    Polynomial
        Remainder of p / q.

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 0.0, 1.0]))  # x^2 + 1
    >>> q = polynomial(torch.tensor([-1.0, 1.0]))  # x - 1
    >>> polynomial_mod(p, q).coeffs  # remainder is 2
    tensor([2.])
    """
    _, remainder = polynomial_divmod(p, q)
    return remainder
