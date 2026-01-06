"""Division of Chebyshev series."""

from __future__ import annotations

from typing import Tuple

from torchscience.polynomial._polynomial import polynomial_divmod

from ._chebyshev_t import ChebyshevT
from ._chebyshev_t_to_polynomial import chebyshev_t_to_polynomial
from ._polynomial_to_chebyshev_t import polynomial_to_chebyshev_t


def chebyshev_t_divmod(
    a: ChebyshevT, b: ChebyshevT
) -> Tuple[ChebyshevT, ChebyshevT]:
    """Divide two Chebyshev series with remainder.

    Returns quotient q and remainder r such that a = b*q + r.

    Parameters
    ----------
    a : ChebyshevT
        Dividend.
    b : ChebyshevT
        Divisor.

    Returns
    -------
    Tuple[ChebyshevT, ChebyshevT]
        (quotient, remainder)

    Notes
    -----
    Converts to power basis, performs division, converts back.

    Examples
    --------
    >>> a = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = chebyshev_t(torch.tensor([1.0, 1.0]))
    >>> q, r = chebyshev_t_divmod(a, b)
    """
    # Convert to power basis
    a_poly = chebyshev_t_to_polynomial(a)
    b_poly = chebyshev_t_to_polynomial(b)

    # Divide in power basis
    q_poly, r_poly = polynomial_divmod(a_poly, b_poly)

    # Convert back to Chebyshev
    q = polynomial_to_chebyshev_t(q_poly)
    r = polynomial_to_chebyshev_t(r_poly)

    return q, r
