from torch import Tensor

import torchscience.ops.torchscience

__all__ = ["kelvin_kei"]


def kelvin_kei(v: Tensor, x: Tensor) -> Tensor:
    r"""
    Kelvin function :math:`\text{kei}_v(x)`.

    The Kelvin function kei is the imaginary part of:

    .. math::
        \text{ker}_v(x) + i \cdot \text{kei}_v(x) = K_v(x \cdot e^{\pi i/4})

    where :math:`K_v` is the modified Bessel function of the second kind.

    Parameters
    ----------
    v : Tensor
        Order of the Kelvin function.
    x : Tensor
        Argument (real, positive).

    Returns
    -------
    Tensor
        The Kelvin function :math:`\text{kei}_v(x)`.
    """
    return torchscience.ops.torchscience._kelvin_kei(v, x)
