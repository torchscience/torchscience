import torch
from torch import Tensor


def log_gamma(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Natural logarithm of the absolute value of the gamma function.

    .. math::

        \\text{log\\_gamma}(x) = \\ln|\\Gamma(x)|

    This is more numerically stable than computing ``log(gamma(x))`` directly,
    especially for large values where gamma would overflow.

    The derivative of log_gamma is the digamma function:

    .. math::

        \\frac{d}{dx} \\ln\\Gamma(x) = \\psi(x)

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The log-gamma function evaluated at each element of `input`.

    Examples
    --------
    >>> log_gamma(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    tensor([0.0000, 0.0000, 0.6931, 1.7918, 3.1781])

    >>> log_gamma(torch.tensor([0.5]))
    tensor([0.5724])
    """
    output: Tensor = torch.ops.torchscience._log_gamma(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
