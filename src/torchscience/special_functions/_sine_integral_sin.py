import torch
from torch import Tensor


def sine_integral_sin(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Shifted sine integral function si(x).

    .. math::

        \\text{si}(x) = \\text{Si}(x) - \\frac{\\pi}{2} = -\\int_x^\\infty \\frac{\\sin t}{t} dt

    This is the sine integral shifted so that si(x) -> 0 as x -> infinity.

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The shifted sine integral evaluated at each element of `input`.

    Examples
    --------
    >>> sine_integral_sin(torch.tensor([0.0, 1.0, 2.0]))
    tensor([-1.5708, -0.6247, -0.1225])

    Notes
    -----
    The shifted sine integral si(x) is related to the standard sine integral
    Si(x) by si(x) = Si(x) - pi/2. This form is sometimes more convenient
    because it has the asymptotic behavior si(x) -> 0 as x -> infinity.
    """
    output: Tensor = torch.ops.torchscience._sine_integral_sin(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
