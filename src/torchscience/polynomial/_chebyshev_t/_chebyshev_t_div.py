"""Quotient of Chebyshev series division."""

from __future__ import annotations

from ._chebyshev_t import ChebyshevT
from ._chebyshev_t_divmod import chebyshev_t_divmod


def chebyshev_t_div(a: ChebyshevT, b: ChebyshevT) -> ChebyshevT:
    """Divide two Chebyshev series, returning quotient only.

    Parameters
    ----------
    a : ChebyshevT
        Dividend.
    b : ChebyshevT
        Divisor.

    Returns
    -------
    ChebyshevT
        Quotient.
    """
    q, _ = chebyshev_t_divmod(a, b)
    return q
