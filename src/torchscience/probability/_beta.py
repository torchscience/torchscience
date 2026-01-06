"""Beta distribution operators."""

import torch
from torch import Tensor

__all__ = ["beta_cdf", "beta_pdf", "beta_ppf"]


def beta_cdf(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    r"""Cumulative distribution function of the beta distribution.

    .. math::
        F(x; a, b) = I_x(a, b)

    where :math:`I_x` is the regularized incomplete beta function.

    Parameters
    ----------
    x : Tensor
        Quantiles in [0, 1].
    a : Tensor
        First shape parameter. Must be positive.
    b : Tensor
        Second shape parameter. Must be positive.

    Returns
    -------
    Tensor
        CDF values.

    Examples
    --------
    >>> x = torch.tensor([0.25, 0.5, 0.75])
    >>> a = torch.tensor(2.0)
    >>> b = torch.tensor(5.0)
    >>> beta_cdf(x, a, b)
    tensor([0.3672, 0.8125, 0.9727])
    """
    return torch.ops.torchscience.beta_cdf(x, a, b)


def beta_pdf(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    r"""Probability density function of the beta distribution.

    .. math::
        f(x; a, b) = \frac{x^{a-1} (1-x)^{b-1}}{B(a, b)}

    where :math:`B(a, b)` is the beta function.

    Parameters
    ----------
    x : Tensor
        Values in (0, 1).
    a : Tensor
        First shape parameter. Must be positive.
    b : Tensor
        Second shape parameter. Must be positive.

    Returns
    -------
    Tensor
        PDF values.
    """
    return torch.ops.torchscience.beta_pdf(x, a, b)


def beta_ppf(p: Tensor, a: Tensor, b: Tensor) -> Tensor:
    r"""Quantile function (inverse CDF) of the beta distribution.

    Parameters
    ----------
    p : Tensor
        Probabilities in [0, 1].
    a : Tensor
        First shape parameter. Must be positive.
    b : Tensor
        Second shape parameter. Must be positive.

    Returns
    -------
    Tensor
        Quantiles.
    """
    return torch.ops.torchscience.beta_ppf(p, a, b)
