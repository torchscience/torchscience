"""Degree of Chebyshev series."""

from __future__ import annotations

import torch
from torch import Tensor

from ._chebyshev_t import ChebyshevT


def chebyshev_t_degree(c: ChebyshevT) -> Tensor:
    """Return degree of Chebyshev series.

    Parameters
    ----------
    c : ChebyshevT
        Chebyshev series.

    Returns
    -------
    Tensor
        Degree (number of coefficients - 1).

    Examples
    --------
    >>> c = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
    >>> chebyshev_t_degree(c)
    tensor(2)
    """
    return torch.tensor(c.coeffs.shape[-1] - 1)
