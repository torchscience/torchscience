import torch
from torch import Tensor


def error_inverse_erf(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Inverse error function.

    .. math::

        \\text{erf}^{-1}(x) = y \\quad \\text{such that} \\quad \\text{erf}(y) = x

    The inverse error function is defined for ``-1 < x < 1``.

    Properties:

    - ``error_inverse_erf(0) = 0``
    - ``error_inverse_erf(-1) = -inf``
    - ``error_inverse_erf(1) = inf``
    - ``error_inverse_erf(-x) = -error_inverse_erf(x)`` (odd function)

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor with values in the interval (-1, 1).

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The inverse error function evaluated at each element of `input`.

    Examples
    --------
    >>> error_inverse_erf(torch.tensor([0.0, 0.5, -0.5]))
    tensor([ 0.0000,  0.4769, -0.4769])

    >>> error_inverse_erf(torch.tensor([0.8427]))  # erf(1) ≈ 0.8427
    tensor([1.0000])
    """
    output: Tensor = torch.ops.torchscience._error_inverse_erf(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
