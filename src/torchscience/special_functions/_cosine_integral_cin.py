import torch
from torch import Tensor


def cosine_integral_cin(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Cosine integral function Cin(x).

    .. math::

        \\text{Cin}(x) = \\int_0^x \\frac{1 - \\cos t}{t} dt

    The function Cin(x) is entire (has no singularities) and is related to the
    cosine integral Ci(x) by:

    .. math::

        \\text{Cin}(x) = \\gamma + \\ln|x| - \\text{Ci}(x)

    where gamma is Euler's constant.

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The Cin function evaluated at each element of `input`.

    Examples
    --------
    >>> cosine_integral_cin(torch.tensor([0.5, 1.0, 2.0]))
    tensor([0.0620, 0.2398, 0.8223])
    """
    output: Tensor = torch.ops.torchscience._cosine_integral_cin(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
