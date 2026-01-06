"""Gamma distribution operators."""

import torch
from torch import Tensor

__all__ = [
    "gamma_cumulative_distribution",
    "gamma_probability_density",
    "gamma_quantile",
]


def gamma_cumulative_distribution(
    x: Tensor, shape: Tensor, scale: Tensor
) -> Tensor:
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
    >>> gamma_cumulative_distribution(x, shape, scale)
    tensor([0.2642, 0.5940, 0.8009])
    """
    return torch.ops.torchscience.gamma_cumulative_distribution(
        x, shape, scale
    )


def gamma_probability_density(
    x: Tensor, shape: Tensor, scale: Tensor
) -> Tensor:
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
    return torch.ops.torchscience.gamma_probability_density(x, shape, scale)


def gamma_quantile(p: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
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
    return torch.ops.torchscience.gamma_quantile(p, shape, scale)
