import torch
from torch import Tensor


def airy_ai(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Airy function of the first kind.

    .. math::

        \\text{Ai}(x) = \\frac{1}{\\pi} \\int_0^\\infty \\cos\\left(\\frac{t^3}{3} + xt\\right) dt

    The Airy function Ai(x) is a solution to the differential equation:

    .. math::

        y'' - xy = 0

    Properties:

    - ``Ai(0) ≈ 0.3550280539``
    - ``Ai(x) -> 0`` as ``x -> +inf`` (exponential decay)
    - ``Ai(x)`` oscillates for ``x < 0``
    - All zeros of ``Ai(x)`` are on the negative real axis

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The Airy function Ai evaluated at each element of `input`.

    Examples
    --------
    >>> airy_ai(torch.tensor([0.0, 1.0, 2.0]))
    tensor([0.3550, 0.1353, 0.0349])

    >>> airy_ai(torch.tensor([-1.0, -2.0]))
    tensor([0.5356, 0.2274])

    Notes
    -----
    The Airy functions arise in many physical problems, including:

    - Quantum mechanics (WKB approximation near turning points)
    - Optics (diffraction patterns)
    - Fluid dynamics (Stokes phenomenon)

    The function is named after George Biddell Airy, who encountered it
    in his study of optics.
    """
    output: Tensor = torch.ops.torchscience._airy_ai(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
