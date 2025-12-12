import torch
from torch import Tensor


def inverse_erfc(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Inverse complementary error function.

    .. math::

        \\text{erfc}^{-1}(x) = y \\quad \\text{such that} \\quad \\text{erfc}(y) = x

    The inverse complementary error function is defined for ``0 < x < 2``.

    Properties:

    - ``inverse_erfc(1) = 0``
    - ``inverse_erfc(0) = inf``
    - ``inverse_erfc(2) = -inf``

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor with values in the interval (0, 2).

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The inverse complementary error function evaluated at each element of `input`.

    Examples
    --------
    >>> inverse_erfc(torch.tensor([0.5, 1.0, 1.5]))
    tensor([ 0.4769,  0.0000, -0.4769])

    >>> inverse_erfc(torch.tensor([0.1, 0.9]))
    tensor([1.1631, 0.0889])
    """
    output: Tensor = torch.ops.torchscience._inverse_erfc(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
