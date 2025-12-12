import torch
from torch import Tensor


def exponential_integral_ei(x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Exponential integral Ei.

    .. math::

        \\mathrm{Ei}(x) = -\\int_{-x}^{\\infty} \\frac{e^{-t}}{t} dt

    Parameters
    ----------
    x : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The exponential integral Ei evaluated at each element.

    Notes
    -----
    The gradient is:

    .. math::

        \\frac{d}{dx} \\mathrm{Ei}(x) = \\frac{e^x}{x}

    Examples
    --------
    >>> exponential_integral_ei(torch.tensor([1.0]))
    tensor([1.8951])

    >>> exponential_integral_ei(torch.tensor([1.0, 2.0, 3.0]))
    tensor([1.8951, 4.9542, 9.9338])
    """
    output: Tensor = torch.ops.torchscience._exponential_integral_ei(x)

    if out is not None:
        out.copy_(output)

        return out

    return output
