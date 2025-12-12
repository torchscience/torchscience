import torch
from torch import Tensor


def erfc(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Complementary error function.

    .. math::

        \\text{erfc}(x) = 1 - \\text{erf}(x) = \\frac{2}{\\sqrt{\\pi}} \\int_x^\\infty e^{-t^2} dt

    The complementary error function is useful for computing ``1 - erf(x)``
    when ``erf(x)`` is close to 1, avoiding catastrophic cancellation.

    Properties:

    - ``erfc(0) = 1``
    - ``erfc(inf) = 0``
    - ``erfc(-inf) = 2``
    - ``erfc(x) + erfc(-x) = 2``

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The complementary error function evaluated at each element of `input`.

    Examples
    --------
    >>> erfc(torch.tensor([0.0, 0.5, 1.0, 2.0]))
    tensor([1.0000, 0.4795, 0.1573, 0.0047])

    >>> erfc(torch.tensor([-1.0, 1.0]))
    tensor([1.8427, 0.1573])
    """
    output: Tensor = torch.ops.torchscience._erfc(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
