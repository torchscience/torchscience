"""Binomial distribution operators."""

import torch
from torch import Tensor

__all__ = ["binomial_cumulative_distribution", "binomial_pmf"]


def binomial_cumulative_distribution(
    k: Tensor, n: Tensor, p: Tensor
) -> Tensor:
    r"""Cumulative distribution function of the binomial distribution.

    .. math::
        F(k; n, p) = P(X \le k) = I_{1-p}(n-k, k+1)

    where :math:`I` is the regularized incomplete beta function.

    Parameters
    ----------
    k : Tensor
        Number of successes. Non-negative integers (floored if float).
    n : Tensor
        Number of trials. Must be positive integer.
    p : Tensor
        Probability of success in [0, 1].

    Returns
    -------
    Tensor
        Cumulative probability :math:`P(X \le k)`.

    Notes
    -----
    Gradients are computed with respect to p only (k and n are discrete).

    Examples
    --------
    >>> k = torch.tensor([0.0, 3.0, 5.0, 10.0])
    >>> n = torch.tensor(10.0)
    >>> p = torch.tensor(0.3)
    >>> binomial_cumulative_distribution(k, n, p)
    tensor([0.0282, 0.6496, 0.9527, 1.0000])
    """
    return torch.ops.torchscience.binomial_cumulative_distribution(k, n, p)


def binomial_pmf(k: Tensor, n: Tensor, p: Tensor) -> Tensor:
    r"""Probability mass function of the binomial distribution.

    .. math::
        P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}

    Parameters
    ----------
    k : Tensor
        Number of successes.
    n : Tensor
        Number of trials.
    p : Tensor
        Probability of success.

    Returns
    -------
    Tensor
        Probability :math:`P(X = k)`.

    Notes
    -----
    Gradients are computed with respect to p only (k and n are discrete).

    Examples
    --------
    >>> k = torch.arange(0, 11, dtype=torch.float32)
    >>> n = torch.tensor(10.0)
    >>> p = torch.tensor(0.3)
    >>> binomial_pmf(k, n, p)
    tensor([0.0282, 0.1211, 0.2335, 0.2668, 0.2001, 0.1029, 0.0368, 0.0090, 0.0014, 0.0001, 0.0000])
    """
    return torch.ops.torchscience.binomial_pmf(k, n, p)
