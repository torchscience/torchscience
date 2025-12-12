from torch import Tensor

import torchscience.ops.torchscience

__all__ = ["kelvin_bei"]


def kelvin_bei(v: Tensor, x: Tensor) -> Tensor:
    r"""
    Kelvin function :math:`\text{bei}_v(x)`.

    The Kelvin function bei is the imaginary part of:

    .. math::
        \text{ber}_v(x) + i \cdot \text{bei}_v(x) = J_v(x \cdot e^{3\pi i/4})

    where :math:`J_v` is the Bessel function of the first kind.

    Parameters
    ----------
    v : Tensor
        Order of the Kelvin function.
    x : Tensor
        Argument (real, non-negative).

    Returns
    -------
    Tensor
        The Kelvin function :math:`\text{bei}_v(x)`.
    """
    return torchscience.ops.torchscience._kelvin_bei(v, x)
