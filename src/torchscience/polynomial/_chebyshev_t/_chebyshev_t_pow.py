"""Raise Chebyshev series to an integer power."""

from __future__ import annotations

import torch

from ._chebyshev_t import ChebyshevT
from ._chebyshev_t_multiply import chebyshev_t_multiply


def chebyshev_t_pow(a: ChebyshevT, n: int) -> ChebyshevT:
    """Raise Chebyshev series to a non-negative integer power.

    Uses binary exponentiation for efficiency.

    Parameters
    ----------
    a : ChebyshevT
        Base series.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    ChebyshevT
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = chebyshev_t(torch.tensor([1.0, 1.0]))  # 1 + T_1
    >>> b = chebyshev_t_pow(a, 2)
    >>> b.coeffs  # (1 + T_1)^2 = 1.5 + 2*T_1 + 0.5*T_2
    tensor([1.5, 2.0, 0.5])
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    if n == 0:
        # a^0 = T_0 = 1
        ones_shape = list(a.coeffs.shape)
        ones_shape[-1] = 1
        return ChebyshevT(
            coeffs=torch.ones(
                ones_shape, dtype=a.coeffs.dtype, device=a.coeffs.device
            )
        )

    if n == 1:
        return ChebyshevT(coeffs=a.coeffs.clone())

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = ChebyshevT(coeffs=base.coeffs.clone())
            else:
                result = chebyshev_t_multiply(result, base)
        base = chebyshev_t_multiply(base, base)
        n //= 2

    return result
