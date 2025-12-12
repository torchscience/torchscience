import torch
from torch import Tensor


def cos_pi(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Cosine of pi times input:

    .. math::

        \\text{cos\\_pi}(x) = \\cos(\\pi x)

    This function computes :math:`\\cos(\\pi x)` more accurately than
    ``torch.cos(torch.pi * x)`` for large values of ``x`` and for values
    near half-integers where the result should be exactly zero.

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        Cosine of pi times each element of the input tensor.

    Examples
    --------
    >>> cos_pi(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0]))
    tensor([ 1.0000,  0.0000, -1.0000,  0.0000,  1.0000])
    """
    output: Tensor = torch.ops.torchscience._cos_pi(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
