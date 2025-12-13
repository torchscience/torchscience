import torch
from torch import Tensor


def euler_number_e(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Computes the Euler number E_n for each element in the input tensor.

    .. math::

        E_n

    The Euler numbers appear in the Taylor series expansion of sec(x):

    .. math::

        \\sec(x) = \\sum_{n=0}^{\\infty} \\frac{(-1)^n E_{2n}}{(2n)!} x^{2n}

    Properties:

    - :math:`E_0 = 1`
    - :math:`E_n = 0` for all odd n
    - Even Euler numbers alternate in sign

    Parameters
    ----------
    input : Tensor
        Input tensor containing non-negative integers n for which E_n is computed.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The Euler number E_n for each element in the input tensor.

    Examples
    --------
    >>> euler_number_e(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    tensor([  1.,   0.,  -1.,   0.,   5.,   0., -61.])
    """
    output: Tensor = torch.ops.torchscience._euler_number_e(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
