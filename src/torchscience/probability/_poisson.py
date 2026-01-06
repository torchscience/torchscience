"""Poisson distribution operators."""

import torch
from torch import Tensor

__all__ = ["poisson_cdf", "poisson_pmf"]


def poisson_cdf(k: Tensor, rate: Tensor) -> Tensor:
    r"""Cumulative distribution function of the Poisson distribution.

    .. math::
        F(k; \lambda) = Q(k+1, \lambda) = \frac{\Gamma(k+1, \lambda)}{\Gamma(k+1)}

    where Q is the upper regularized incomplete gamma function.

    Parameters
    ----------
    k : Tensor
        Number of events. Non-negative integers (floored if float).
    rate : Tensor
        Rate parameter :math:`\lambda` (mean). Must be positive.

    Returns
    -------
    Tensor
        Cumulative probability :math:`P(X \le k)`.

    Notes
    -----
    Gradients are computed with respect to rate only (k is discrete).

    Examples
    --------
    >>> k = torch.tensor([0.0, 2.0, 5.0, 10.0])
    >>> rate = torch.tensor(5.0)
    >>> poisson_cdf(k, rate)
    tensor([0.0067, 0.1247, 0.6160, 0.9863])
    """
    return torch.ops.torchscience.poisson_cdf(k, rate)


def poisson_pmf(k: Tensor, rate: Tensor) -> Tensor:
    r"""Probability mass function of the Poisson distribution.

    .. math::
        P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}

    Parameters
    ----------
    k : Tensor
        Number of events.
    rate : Tensor
        Rate parameter.

    Returns
    -------
    Tensor
        Probability :math:`P(X = k)`.

    Notes
    -----
    Gradients are computed with respect to rate only (k is discrete).

    Examples
    --------
    >>> k = torch.arange(0, 10, dtype=torch.float32)
    >>> rate = torch.tensor(5.0)
    >>> poisson_pmf(k, rate)
    tensor([0.0067, 0.0337, 0.0842, 0.1404, 0.1755, 0.1755, 0.1462, 0.1044, 0.0653, 0.0363])
    """
    return torch.ops.torchscience.poisson_pmf(k, rate)
