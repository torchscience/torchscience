"""Normal distribution operators."""

import torch
from torch import Tensor

__all__ = [
    "normal_cumulative_distribution",
    "normal_probability_density",
    "normal_quantile",
    "normal_survival",
    "normal_logpdf",
]


def normal_cumulative_distribution(
    x: Tensor,
    loc: Tensor | float = 0.0,
    scale: Tensor | float = 1.0,
) -> Tensor:
    r"""Cumulative distribution function of the normal distribution.

    .. math::
        \Phi(x; \mu, \sigma) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x - \mu}{\sigma \sqrt{2}}\right)\right]

    Parameters
    ----------
    x : Tensor
        Quantiles at which to evaluate the CDF.
    loc : Tensor or float, default=0.0
        Mean (location parameter) :math:`\mu`.
    scale : Tensor or float, default=1.0
        Standard deviation (scale parameter) :math:`\sigma`. Must be positive.

    Returns
    -------
    Tensor
        CDF values :math:`P(X \le x)` where :math:`X \sim \mathcal{N}(\mu, \sigma^2)`.

    Examples
    --------
    >>> x = torch.tensor([0.0, 1.0, 2.0])
    >>> normal_cumulative_distribution(x)
    tensor([0.5000, 0.8413, 0.9772])

    >>> normal_cumulative_distribution(x, loc=1.0, scale=0.5)
    tensor([0.0228, 0.5000, 0.9772])

    Notes
    -----
    The gradient with respect to x equals the PDF:

    .. math::
        \frac{\partial \Phi}{\partial x} = \phi(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

    See Also
    --------
    normal_probability_density : Probability density function
    normal_quantile : Inverse CDF (quantile function)
    normal_survival : Survival function (1 - CDF)
    """
    loc_t = (
        loc
        if isinstance(loc, Tensor)
        else torch.as_tensor(loc, dtype=x.dtype, device=x.device)
    )
    scale_t = (
        scale
        if isinstance(scale, Tensor)
        else torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.normal_cumulative_distribution(
        x, loc_t, scale_t
    )


def normal_probability_density(
    x: Tensor,
    loc: Tensor | float = 0.0,
    scale: Tensor | float = 1.0,
) -> Tensor:
    r"""Probability density function of the normal distribution.

    .. math::
        \phi(x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the PDF.
    loc : Tensor or float, default=0.0
        Mean parameter.
    scale : Tensor or float, default=1.0
        Standard deviation parameter. Must be positive.

    Returns
    -------
    Tensor
        PDF values.

    Examples
    --------
    >>> x = torch.tensor([0.0])
    >>> normal_probability_density(x)  # Peak of standard normal
    tensor([0.3989])

    See Also
    --------
    normal_logpdf : Log PDF (more numerically stable)
    normal_cumulative_distribution : Cumulative distribution function
    """
    loc_t = (
        loc
        if isinstance(loc, Tensor)
        else torch.as_tensor(loc, dtype=x.dtype, device=x.device)
    )
    scale_t = (
        scale
        if isinstance(scale, Tensor)
        else torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.normal_probability_density(x, loc_t, scale_t)


def normal_quantile(
    p: Tensor,
    loc: Tensor | float = 0.0,
    scale: Tensor | float = 1.0,
) -> Tensor:
    r"""Percent point function (quantile function, inverse CDF) of the normal distribution.

    Returns x such that :math:`P(X \le x) = p` for :math:`X \sim \mathcal{N}(\mu, \sigma^2)`.

    .. math::
        \Phi^{-1}(p; \mu, \sigma) = \mu + \sigma \cdot \Phi^{-1}(p)

    Parameters
    ----------
    p : Tensor
        Probability values in (0, 1).
    loc : Tensor or float, default=0.0
        Mean parameter.
    scale : Tensor or float, default=1.0
        Standard deviation parameter. Must be positive.

    Returns
    -------
    Tensor
        Quantile values.

    Examples
    --------
    >>> p = torch.tensor([0.025, 0.5, 0.975])
    >>> normal_quantile(p)
    tensor([-1.9600,  0.0000,  1.9600])

    See Also
    --------
    normal_cumulative_distribution : Inverse of PPF
    """
    loc_t = (
        loc
        if isinstance(loc, Tensor)
        else torch.as_tensor(loc, dtype=p.dtype, device=p.device)
    )
    scale_t = (
        scale
        if isinstance(scale, Tensor)
        else torch.as_tensor(scale, dtype=p.dtype, device=p.device)
    )
    return torch.ops.torchscience.normal_quantile(p, loc_t, scale_t)


def normal_survival(
    x: Tensor,
    loc: Tensor | float = 0.0,
    scale: Tensor | float = 1.0,
) -> Tensor:
    r"""Survival function (1 - CDF) of the normal distribution.

    .. math::
        S(x) = P(X > x) = 1 - \Phi(x) = \frac{1}{2} \text{erfc}\left(\frac{x - \mu}{\sigma \sqrt{2}}\right)

    More numerically stable than ``1 - normal_cumulative_distribution(x)`` for large x.

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the survival function.
    loc : Tensor or float, default=0.0
        Mean parameter.
    scale : Tensor or float, default=1.0
        Standard deviation parameter. Must be positive.

    Returns
    -------
    Tensor
        Survival function values.

    Examples
    --------
    >>> x = torch.tensor([0.0, 1.0, 2.0])
    >>> normal_survival(x)
    tensor([0.5000, 0.1587, 0.0228])

    See Also
    --------
    normal_cumulative_distribution : CDF = 1 - SF
    """
    loc_t = (
        loc
        if isinstance(loc, Tensor)
        else torch.as_tensor(loc, dtype=x.dtype, device=x.device)
    )
    scale_t = (
        scale
        if isinstance(scale, Tensor)
        else torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.normal_survival(x, loc_t, scale_t)


def normal_logpdf(
    x: Tensor,
    loc: Tensor | float = 0.0,
    scale: Tensor | float = 1.0,
) -> Tensor:
    r"""Log of the probability density function of the normal distribution.

    .. math::
        \log \phi(x) = -\frac{1}{2}\log(2\pi) - \log(\sigma) - \frac{(x-\mu)^2}{2\sigma^2}

    More numerically stable than ``log(normal_probability_density(x))`` for extreme x.

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the log PDF.
    loc : Tensor or float, default=0.0
        Mean parameter.
    scale : Tensor or float, default=1.0
        Standard deviation parameter. Must be positive.

    Returns
    -------
    Tensor
        Log PDF values.

    Examples
    --------
    >>> x = torch.tensor([0.0])
    >>> normal_logpdf(x)  # log(0.3989) = -0.9189
    tensor([-0.9189])

    See Also
    --------
    normal_probability_density : Exp of log PDF
    """
    loc_t = (
        loc
        if isinstance(loc, Tensor)
        else torch.as_tensor(loc, dtype=x.dtype, device=x.device)
    )
    scale_t = (
        scale
        if isinstance(scale, Tensor)
        else torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.normal_logpdf(x, loc_t, scale_t)
