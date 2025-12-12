import torch
from torch import Tensor


def trigamma(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Trigamma function.

    The trigamma function is the second derivative of the log-gamma function,
    or equivalently the derivative of the digamma function:

    .. math::

        \\psi_1(x) = \\frac{d^2}{dx^2} \\ln\\Gamma(x) = \\frac{d}{dx} \\psi(x)

    It can also be expressed as an infinite series:

    .. math::

        \\psi_1(x) = \\sum_{k=0}^{\\infty} \\frac{1}{(x+k)^2}

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The trigamma function evaluated at each element of `input`.

    Examples
    --------
    >>> trigamma(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    tensor([1.6449, 0.6449, 0.3949, 0.2838])

    >>> trigamma(torch.tensor([0.5]))
    tensor([4.9348])
    """
    output: Tensor = torch.ops.torchscience._trigamma(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
