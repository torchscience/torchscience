"""F distribution operators."""

import torch
from torch import Tensor

__all__ = [
    "f_cumulative_distribution",
    "f_probability_density",
    "f_quantile",
    "f_survival",
]


def f_cumulative_distribution(
    x: Tensor, dfn: Tensor | float, dfd: Tensor | float
) -> Tensor:
    r"""Cumulative distribution function of the F-distribution.

    .. math::
        F(x; d_1, d_2) = I_{x'}(d_1/2, d_2/2)

    where :math:`x' = \frac{d_1 x}{d_1 x + d_2}` and :math:`I_x(a,b)` is the
    regularized incomplete beta function.

    Parameters
    ----------
    x : Tensor
        Quantiles. Must be non-negative.
    dfn : Tensor or float
        Numerator degrees of freedom :math:`d_1`. Must be positive.
    dfd : Tensor or float
        Denominator degrees of freedom :math:`d_2`. Must be positive.

    Returns
    -------
    Tensor
        CDF values :math:`P(X \le x)` where :math:`X \sim F(d_1, d_2)`.

    Examples
    --------
    >>> x = torch.tensor([0.5, 1.0, 2.0])
    >>> f_cumulative_distribution(x, dfn=5.0, dfd=10.0)
    tensor([0.2211, 0.5000, 0.8406])

    Notes
    -----
    When :math:`d_1 = d_2`, the CDF at :math:`x = 1` equals 0.5 (the median).

    The F-distribution arises as the ratio of two independent chi-squared
    random variables divided by their degrees of freedom:

    .. math::
        F = \frac{X_1 / d_1}{X_2 / d_2}

    where :math:`X_1 \sim \chi^2(d_1)` and :math:`X_2 \sim \chi^2(d_2)`.

    See Also
    --------
    f_probability_density : Probability density function
    f_quantile : Inverse CDF (quantile function)
    f_survival : Survival function (1 - CDF)
    """
    dfn_t = (
        dfn
        if isinstance(dfn, Tensor)
        else torch.as_tensor(dfn, dtype=x.dtype, device=x.device)
    )
    dfd_t = (
        dfd
        if isinstance(dfd, Tensor)
        else torch.as_tensor(dfd, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.f_cumulative_distribution(x, dfn_t, dfd_t)


def f_probability_density(
    x: Tensor, dfn: Tensor | float, dfd: Tensor | float
) -> Tensor:
    r"""Probability density function of the F-distribution.

    .. math::
        f(x; d_1, d_2) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1+d_2}}}}{x \, B(d_1/2, d_2/2)}

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the PDF. Must be non-negative.
    dfn : Tensor or float
        Numerator degrees of freedom :math:`d_1`. Must be positive.
    dfd : Tensor or float
        Denominator degrees of freedom :math:`d_2`. Must be positive.

    Returns
    -------
    Tensor
        PDF values.

    Examples
    --------
    >>> x = torch.linspace(0.1, 3, 5)
    >>> f_probability_density(x, dfn=5.0, dfd=10.0)
    tensor([0.2156, 0.6694, 0.4579, 0.2013, 0.0719])

    Notes
    -----
    The mode occurs at :math:`x = \frac{d_1 - 2}{d_1} \cdot \frac{d_2}{d_2 + 2}`
    for :math:`d_1 > 2`.

    See Also
    --------
    f_cumulative_distribution : Cumulative distribution function
    """
    dfn_t = (
        dfn
        if isinstance(dfn, Tensor)
        else torch.as_tensor(dfn, dtype=x.dtype, device=x.device)
    )
    dfd_t = (
        dfd
        if isinstance(dfd, Tensor)
        else torch.as_tensor(dfd, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.f_probability_density(x, dfn_t, dfd_t)


def f_quantile(p: Tensor, dfn: Tensor | float, dfd: Tensor | float) -> Tensor:
    r"""Percent point function (quantile function) of the F-distribution.

    Returns :math:`x` such that :math:`P(X \le x) = p` for :math:`X \sim F(d_1, d_2)`.

    Parameters
    ----------
    p : Tensor
        Probability values in (0, 1).
    dfn : Tensor or float
        Numerator degrees of freedom :math:`d_1`. Must be positive.
    dfd : Tensor or float
        Denominator degrees of freedom :math:`d_2`. Must be positive.

    Returns
    -------
    Tensor
        Quantile values.

    Examples
    --------
    >>> p = torch.tensor([0.05, 0.5, 0.95])
    >>> f_quantile(p, dfn=5.0, dfd=10.0)
    tensor([0.2621, 0.9356, 3.3258])

    Notes
    -----
    The implementation uses bisection followed by Newton-Raphson refinement
    for robust convergence.

    See Also
    --------
    f_cumulative_distribution : Inverse of PPF
    """
    dfn_t = (
        dfn
        if isinstance(dfn, Tensor)
        else torch.as_tensor(dfn, dtype=p.dtype, device=p.device)
    )
    dfd_t = (
        dfd
        if isinstance(dfd, Tensor)
        else torch.as_tensor(dfd, dtype=p.dtype, device=p.device)
    )
    return torch.ops.torchscience.f_quantile(p, dfn_t, dfd_t)


def f_survival(x: Tensor, dfn: Tensor | float, dfd: Tensor | float) -> Tensor:
    r"""Survival function (1 - CDF) of the F-distribution.

    .. math::
        S(x; d_1, d_2) = 1 - F(x; d_1, d_2) = I_{1-x'}(d_2/2, d_1/2)

    where :math:`1 - x' = \frac{d_2}{d_1 x + d_2}`.

    More numerically stable than ``1 - f_cumulative_distribution(x, dfn, dfd)`` for large x.

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the survival function.
    dfn : Tensor or float
        Numerator degrees of freedom :math:`d_1`. Must be positive.
    dfd : Tensor or float
        Denominator degrees of freedom :math:`d_2`. Must be positive.

    Returns
    -------
    Tensor
        Survival function values :math:`P(X > x)`.

    Examples
    --------
    >>> x = torch.tensor([0.5, 1.0, 2.0])
    >>> f_survival(x, dfn=5.0, dfd=10.0)
    tensor([0.7789, 0.5000, 0.1594])

    See Also
    --------
    f_cumulative_distribution : CDF = 1 - SF
    """
    dfn_t = (
        dfn
        if isinstance(dfn, Tensor)
        else torch.as_tensor(dfn, dtype=x.dtype, device=x.device)
    )
    dfd_t = (
        dfd
        if isinstance(dfd, Tensor)
        else torch.as_tensor(dfd, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.f_survival(x, dfn_t, dfd_t)
