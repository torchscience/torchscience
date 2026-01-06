"""Find roots of Chebyshev series."""

from __future__ import annotations

import torch
from torch import Tensor

from ._chebyshev_t import ChebyshevT
from ._chebyshev_t_companion import chebyshev_t_companion


def chebyshev_t_roots(c: ChebyshevT) -> Tensor:
    """Find roots of Chebyshev series.

    Computes roots as eigenvalues of the companion matrix.

    Parameters
    ----------
    c : ChebyshevT
        Chebyshev series.

    Returns
    -------
    Tensor
        Complex tensor of roots, shape (n,) where n = degree.

    Notes
    -----
    Uses eigenvalue decomposition of the Chebyshev companion matrix.
    Roots are returned as complex numbers even if all roots are real.

    Examples
    --------
    >>> c = chebyshev_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
    >>> roots = chebyshev_t_roots(c)
    >>> roots.real.sort().values
    tensor([-0.7071,  0.7071])
    """
    A = chebyshev_t_companion(c)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(A)

    return eigenvalues
