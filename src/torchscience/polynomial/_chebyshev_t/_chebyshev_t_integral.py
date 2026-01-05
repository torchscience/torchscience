"""Definite integral of Chebyshev series."""

from __future__ import annotations

from torch import Tensor

from ._chebyshev_t import ChebyshevT
from ._chebyshev_t_antiderivative import chebyshev_t_antiderivative
from ._chebyshev_t_evaluate import chebyshev_t_evaluate


def chebyshev_t_integral(
    a: ChebyshevT, lower: Tensor, upper: Tensor
) -> Tensor:
    """Compute definite integral of Chebyshev series.

    Computes integral_{lower}^{upper} a(x) dx.

    Parameters
    ----------
    a : ChebyshevT
        Series to integrate.
    lower : Tensor
        Lower limit of integration.
    upper : Tensor
        Upper limit of integration.

    Returns
    -------
    Tensor
        Value of definite integral.

    Notes
    -----
    Computed as F(upper) - F(lower) where F is the antiderivative.

    Examples
    --------
    >>> a = chebyshev_t(torch.tensor([1.0]))  # constant 1
    >>> chebyshev_t_integral(a, torch.tensor(-1.0), torch.tensor(1.0))
    tensor(2.)
    """
    # Compute antiderivative with C=0
    antideriv = chebyshev_t_antiderivative(a, constant=0.0)

    # Evaluate at endpoints
    f_upper = chebyshev_t_evaluate(antideriv, upper)
    f_lower = chebyshev_t_evaluate(antideriv, lower)

    return f_upper - f_lower
