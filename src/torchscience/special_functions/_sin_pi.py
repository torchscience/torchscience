import torch
from torch import Tensor


def sin_pi(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Sine of pi times input:

    .. math::

        \\text{sin\\_pi}(x) = \\sin(\\pi x)

    This function computes :math:`\\sin(\\pi x)` more accurately than
    ``torch.sin(torch.pi * x)`` for large values of ``x`` and for values
    near integers where the result should be exactly zero.

    Parameters
    ----------
    input : Tensor, shape=(...,)
        Input tensor.

    out : Tensor, shape=(...,), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...,)
        Sine of pi times each element of the input tensor.

    Examples
    --------
    >>> sin_pi(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0]))
    tensor([ 0.0000,  1.0000,  0.0000, -1.0000,  0.0000])
    """
    output: Tensor = torch.ops.torchscience._sin_pi(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
