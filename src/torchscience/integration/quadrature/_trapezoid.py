"""Trapezoidal rule for numerical integration."""

from typing import Optional

import torch
from torch import Tensor


def trapezoid(
    y: Tensor,
    x: Optional[Tensor] = None,
    *,
    dx: Optional[float] = None,
    dim: int = -1,
) -> Tensor:
    """
    Integrate y along the given dimension using the composite trapezoidal rule.

    Parameters
    ----------
    y : Tensor
        Values to integrate.
    x : Tensor, optional
        Sample points. If None, uses uniform spacing with ``dx``.
    dx : float, optional
        Spacing between sample points when x is None. Default is 1.0 if neither
        x nor dx is specified.
    dim : int
        Dimension along which to integrate.

    Returns
    -------
    Tensor
        Definite integral approximation. Shape is y.shape with ``dim`` removed.

    Notes
    -----
    Fully differentiable with respect to both ``y`` and ``x``.

    Examples
    --------
    >>> y = torch.sin(torch.linspace(0, torch.pi, 100))
    >>> trapezoid(y, dx=torch.pi / 99)  # approximately 2.0
    """
    if x is not None:
        return torch.trapezoid(y, x, dim=dim)
    elif dx is not None:
        return torch.trapezoid(y, dx=dx, dim=dim)
    else:
        return torch.trapezoid(y, dim=dim)
