import torch
from torch import Tensor


def beta(a: Tensor, b: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    .. math::

        \\text{beta}(a, b) = B(a, b) = \\frac{\\Gamma(a) \\Gamma(b)}{\\Gamma(a + b)}

    where :math:`\\Gamma` is the gamma function.

    Parameters
    ----------
    a : Tensor, shape=(...)
        First input tensor.

    b : Tensor, shape=(...)
        Second input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The beta function evaluated at each element pair of `a` and `b`.

    Examples
    --------
    >>> beta(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
    tensor([1.0000, 0.1667, 0.0333])

    >>> beta(torch.tensor([0.5]), torch.tensor([0.5]))
    tensor([3.1416])
    """
    output: Tensor = torch.ops.torchscience._beta(a, b)

    if out is not None:
        out.copy_(output)

        return out

    return output
