import torch
from torch import Tensor


def cosine_integral_ci(x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Cosine integral Ci.

    .. math::

        \\mathrm{Ci}(x) = \\gamma + \\ln|x| + \\int_0^x \\frac{\\cos(t) - 1}{t} dt

    where :math:`\\gamma` is the Euler-Mascheroni constant.

    Parameters
    ----------
    x : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The cosine integral Ci evaluated at each element.

    Notes
    -----
    The gradient is:

    .. math::

        \\frac{d}{dx} \\mathrm{Ci}(x) = \\frac{\\cos(x)}{x}

    Examples
    --------
    >>> cosine_integral_ci(torch.tensor([1.0]))
    tensor([0.3374])

    >>> cosine_integral_ci(torch.tensor([1.0, 2.0, 3.0]))
    tensor([ 0.3374, 0.4230, 0.1196])
    """
    output: Tensor = torch.ops.torchscience._cosine_integral_ci(x)

    if out is not None:
        out.copy_(output)

        return out

    return output
