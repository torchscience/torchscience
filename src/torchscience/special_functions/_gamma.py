import torch
from torch import Tensor


def gamma(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    .. math::

        \\text{gamma}(x) = \\Gamma(x) = \\int_0^\\infty t^{x-1} e^{-t} dt

    For positive integers, :math:`\\Gamma(n) = (n-1)!`.

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The gamma function evaluated at each element of `input`.

    Examples
    --------
    >>> gamma(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    tensor([ 1.,  1.,  2.,  6., 24.])

    >>> gamma(torch.tensor([0.5]))
    tensor([1.7725])
    """
    output: Tensor = torch.ops.torchscience._gamma(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
