import torch
from torch import Tensor


def airy_bi(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Airy function of the second kind.

    .. math::

        \\text{Bi}(x) = \\frac{1}{\\pi} \\int_0^\\infty \\left[
            \\exp\\left(-\\frac{t^3}{3} + xt\\right) +
            \\sin\\left(\\frac{t^3}{3} + xt\\right)
        \\right] dt

    The Airy function Bi(x) is a solution to the differential equation:

    .. math::

        y'' - xy = 0

    that is linearly independent from Ai(x).

    Properties:

    - ``Bi(0) ≈ 0.6149266274``
    - ``Bi(x) -> +inf`` as ``x -> +inf`` (exponential growth)
    - ``Bi(x)`` oscillates for ``x < 0``
    - All zeros of ``Bi(x)`` are on the negative real axis

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The Airy function Bi evaluated at each element of `input`.

    Examples
    --------
    >>> airy_bi(torch.tensor([0.0, 1.0, 2.0]))
    tensor([0.6149, 1.2074, 3.2981])

    >>> airy_bi(torch.tensor([-1.0, -2.0]))
    tensor([0.1040, 0.4123])

    Notes
    -----
    The Airy functions Ai and Bi form a fundamental set of solutions to
    the Airy differential equation. Unlike Ai(x) which decays exponentially
    for positive x, Bi(x) grows exponentially.

    The Wronskian of Ai and Bi is:

    .. math::

        W(\\text{Ai}, \\text{Bi}) = \\text{Ai}(x)\\text{Bi}'(x) -
            \\text{Ai}'(x)\\text{Bi}(x) = \\frac{1}{\\pi}
    """
    output: Tensor = torch.ops.torchscience._airy_bi(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
