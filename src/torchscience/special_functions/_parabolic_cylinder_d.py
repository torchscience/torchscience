from torch import Tensor

import torchscience.ops.torchscience


def parabolic_cylinder_d(nu: Tensor, z: Tensor) -> Tensor:
    r"""
    Parabolic cylinder function :math:`D_{\nu}(z)`.

    The parabolic cylinder function is a solution to Weber's differential equation:

    .. math::

        \frac{d^2 y}{dz^2} + \left(\nu + \frac{1}{2} - \frac{z^2}{4}\right) y = 0

    It can be expressed in terms of the confluent hypergeometric function as:

    .. math::

        D_{\nu}(z) = 2^{\nu/2} e^{-z^2/4} U\left(-\frac{\nu}{2}, \frac{1}{2}, \frac{z^2}{2}\right)

    where :math:`U(a, b, z)` is Tricomi's confluent hypergeometric function.

    Parameters
    ----------
    nu : Tensor
        Order parameter :math:`\nu`.
    z : Tensor
        Argument.

    Returns
    -------
    Tensor
        Value of the parabolic cylinder function :math:`D_{\nu}(z)`.
    """
    return torchscience.ops.torchscience._parabolic_cylinder_d(nu, z)
