"""Remainder of Chebyshev series division."""

from __future__ import annotations

from ._chebyshev_t import ChebyshevT
from ._chebyshev_t_divmod import chebyshev_t_divmod


def chebyshev_t_mod(a: ChebyshevT, b: ChebyshevT) -> ChebyshevT:
    """Divide two Chebyshev series, returning remainder only.

    Parameters
    ----------
    a : ChebyshevT
        Dividend.
    b : ChebyshevT
        Divisor.

    Returns
    -------
    ChebyshevT
        Remainder.
    """
    _, r = chebyshev_t_divmod(a, b)
    return r
