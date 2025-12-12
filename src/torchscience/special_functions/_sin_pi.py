import torch
from torch import Tensor


def sin_pi(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    .. math::

        \\text{sin\\_pi}(x) = \\sin(\\pi x)

    Parameters
    ----------
    input : Tensor, shape=(...,)

    out : Tensor, shape=(...,), optional

    Returns
    -------
    Tensor, shape=(...,)

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
