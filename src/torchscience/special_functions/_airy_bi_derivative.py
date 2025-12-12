import torch
from torch import Tensor


def airy_bi_derivative(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Derivative of the Airy function Bi.

    .. math::

        \\text{Bi}'(x) = \\frac{d}{dx} \\text{Bi}(x)

    The Airy functions satisfy the differential equation y'' = xy,
    which gives Bi''(x) = x * Bi(x).

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The derivative of Bi evaluated at each element of `input`.

    Examples
    --------
    >>> airy_bi_derivative(torch.tensor([0.0, 1.0, 2.0]))
    tensor([0.4483, 0.9324, 3.2981])
    """
    output: Tensor = torch.ops.torchscience._airy_bi_derivative(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
