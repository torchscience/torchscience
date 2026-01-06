"""Chi-squared distribution operators."""

import torch
from torch import Tensor

__all__ = [
    "chi2_cumulative_distribution",
    "chi2_probability_density",
    "chi2_quantile",
    "chi2_survival",
]


def chi2_cumulative_distribution(x: Tensor, df: Tensor | float) -> Tensor:
    r"""Cumulative distribution function of the chi-squared distribution.

    .. math::
        F(x; k) = P(k/2, x/2) = \frac{\gamma(k/2, x/2)}{\Gamma(k/2)}

    where :math:`P(a, x)` is the regularized lower incomplete gamma function.

    Parameters
    ----------
    x : Tensor
        Quantiles. Must be non-negative.
    df : Tensor or float
        Degrees of freedom :math:`k`. Must be positive.

    Returns
    -------
    Tensor
        CDF values :math:`P(X \le x)` where :math:`X \sim \chi^2(k)`.

    Examples
    --------
    >>> x = torch.tensor([1.0, 5.0, 10.0])
    >>> df = torch.tensor(5.0)
    >>> chi2_cumulative_distribution(x, df)
    tensor([0.0374, 0.5841, 0.9247])

    Notes
    -----
    For :math:`k = 2`, the chi-squared distribution reduces to the exponential
    distribution with rate :math:`\lambda = 1/2`:

    .. math::
        F(x; 2) = 1 - e^{-x/2}

    See Also
    --------
    chi2_probability_density : Probability density function
    chi2_quantile : Inverse CDF (quantile function)
    chi2_survival : Survival function (1 - CDF)
    """
    df_t = (
        df
        if isinstance(df, Tensor)
        else torch.as_tensor(df, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.chi2_cumulative_distribution(x, df_t)


def chi2_probability_density(x: Tensor, df: Tensor | float) -> Tensor:
    r"""Probability density function of the chi-squared distribution.

    .. math::
        f(x; k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2-1} e^{-x/2}

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the PDF. Must be non-negative.
    df : Tensor or float
        Degrees of freedom :math:`k`. Must be positive.

    Returns
    -------
    Tensor
        PDF values.

    Examples
    --------
    >>> x = torch.linspace(0, 10, 5)
    >>> chi2_probability_density(x, df=3.0)
    tensor([0.0000, 0.2420, 0.1353, 0.0519, 0.0167])

    Notes
    -----
    The PDF has a mode at :math:`x = k - 2` for :math:`k \ge 2`, and at
    :math:`x = 0` for :math:`k < 2`.

    See Also
    --------
    chi2_cumulative_distribution : Cumulative distribution function
    """
    df_t = (
        df
        if isinstance(df, Tensor)
        else torch.as_tensor(df, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.chi2_probability_density(x, df_t)


def chi2_quantile(p: Tensor, df: Tensor | float) -> Tensor:
    r"""Percent point function (quantile function) of the chi-squared distribution.

    Returns :math:`x` such that :math:`P(X \le x) = p` for :math:`X \sim \chi^2(k)`.

    Parameters
    ----------
    p : Tensor
        Probability values in (0, 1).
    df : Tensor or float
        Degrees of freedom :math:`k`. Must be positive.

    Returns
    -------
    Tensor
        Quantile values.

    Examples
    --------
    >>> p = torch.tensor([0.05, 0.5, 0.95])
    >>> chi2_quantile(p, df=5.0)
    tensor([ 1.1455,  4.3515, 11.0705])

    Notes
    -----
    The implementation uses Wilson-Hilferty approximation for the initial guess
    followed by Newton-Raphson refinement.

    See Also
    --------
    chi2_cumulative_distribution : Inverse of PPF
    """
    df_t = (
        df
        if isinstance(df, Tensor)
        else torch.as_tensor(df, dtype=p.dtype, device=p.device)
    )
    return torch.ops.torchscience.chi2_quantile(p, df_t)


def chi2_survival(x: Tensor, df: Tensor | float) -> Tensor:
    r"""Survival function (1 - CDF) of the chi-squared distribution.

    .. math::
        S(x; k) = 1 - F(x; k) = Q(k/2, x/2)

    where :math:`Q(a, x)` is the regularized upper incomplete gamma function.

    More numerically stable than ``1 - chi2_cumulative_distribution(x, df)`` for large x.

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the survival function.
    df : Tensor or float
        Degrees of freedom :math:`k`. Must be positive.

    Returns
    -------
    Tensor
        Survival function values :math:`P(X > x)`.

    Examples
    --------
    >>> x = torch.tensor([1.0, 5.0, 10.0])
    >>> chi2_survival(x, df=5.0)
    tensor([0.9626, 0.4159, 0.0753])

    See Also
    --------
    chi2_cumulative_distribution : CDF = 1 - SF
    """
    df_t = (
        df
        if isinstance(df, Tensor)
        else torch.as_tensor(df, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.chi2_survival(x, df_t)
