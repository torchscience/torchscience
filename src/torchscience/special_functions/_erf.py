import torch
from torch import Tensor


def erf(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Gauss error function.

    .. math::

        \\text{erf}(x) = \\frac{2}{\\sqrt{\\pi}} \\int_0^x e^{-t^2} dt

    The error function is an odd function with the following properties:

    - ``erf(0) = 0``
    - ``erf(inf) = 1``
    - ``erf(-inf) = -1``
    - ``erf(-x) = -erf(x)``

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The error function evaluated at each element of `input`.

    Examples
    --------
    >>> erf(torch.tensor([0.0, 0.5, 1.0, 2.0]))
    tensor([0.0000, 0.5205, 0.8427, 0.9953])

    >>> erf(torch.tensor([-1.0, 1.0]))
    tensor([-0.8427,  0.8427])
    """
    output: Tensor = torch.ops.torchscience._erf(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
