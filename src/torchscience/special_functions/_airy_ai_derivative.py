import torch
from torch import Tensor


def airy_ai_derivative(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Derivative of the Airy function Ai.

    .. math::

        \\text{Ai}'(x) = \\frac{d}{dx} \\text{Ai}(x)

    The Airy functions satisfy the differential equation y'' = xy,
    which gives Ai''(x) = x * Ai(x).

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The derivative of Ai evaluated at each element of `input`.

    Examples
    --------
    >>> airy_ai_derivative(torch.tensor([0.0, 1.0, 2.0]))
    tensor([-0.2588, -0.1591, -0.0530])
    """
    output: Tensor = torch.ops.torchscience._airy_ai_derivative(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
