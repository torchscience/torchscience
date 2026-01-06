"""Gamma distribution operators."""

import torch
from torch import Tensor

__all__ = ["gamma_cdf", "gamma_pdf", "gamma_ppf"]


def gamma_cdf(x: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    r"""Cumulative distribution function of the gamma distribution.

    .. math::
        F(x; k, \theta) = P(k, x/\theta)

    where :math:`P(a, x)` is the regularized lower incomplete gamma function.

    Parameters
    ----------
    x : Tensor
        Quantiles. Must be non-negative.
    shape : Tensor
        Shape parameter k (or alpha). Must be positive.
    scale : Tensor
        Scale parameter theta. Must be positive.

    Returns
    -------
    Tensor
        CDF values.

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> shape = torch.tensor(2.0)
    >>> scale = torch.tensor(1.0)
    >>> gamma_cdf(x, shape, scale)
    tensor([0.2642, 0.5940, 0.8009])
    """
    return torch.ops.torchscience.gamma_cdf(x, shape, scale)


def gamma_pdf(x: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    r"""Probability density function of the gamma distribution.

    .. math::
        f(x; k, \theta) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}

    Parameters
    ----------
    x : Tensor
        Values. Must be non-negative.
    shape : Tensor
        Shape parameter k (or alpha). Must be positive.
    scale : Tensor
        Scale parameter theta. Must be positive.

    Returns
    -------
    Tensor
        PDF values.
    """
    return torch.ops.torchscience.gamma_pdf(x, shape, scale)


def gamma_ppf(p: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    r"""Quantile function (inverse CDF) of the gamma distribution.

    Parameters
    ----------
    p : Tensor
        Probabilities in [0, 1].
    shape : Tensor
        Shape parameter k (or alpha). Must be positive.
    scale : Tensor
        Scale parameter theta. Must be positive.

    Returns
    -------
    Tensor
        Quantiles.
    """
    return torch.ops.torchscience.gamma_ppf(p, shape, scale)
