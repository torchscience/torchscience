from torch import Tensor

import torchscience.ops.torchscience

__all__ = ["riemann_zeta"]


def riemann_zeta(input: Tensor) -> Tensor:
    r"""
    Riemann zeta function :math:`\zeta(s)`.

    The Riemann zeta function is defined for :math:`\text{Re}(s) > 1` by:

    .. math::
        \zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}

    and is analytically continued to other values of :math:`s`.

    Special values include:

    - :math:`\zeta(2) = \frac{\pi^2}{6}`
    - :math:`\zeta(4) = \frac{\pi^4}{90}`
    - :math:`\zeta(-1) = -\frac{1}{12}`

    The function has a simple pole at :math:`s = 1`.

    Parameters
    ----------
    input : Tensor
        Input tensor (the argument s).

    Returns
    -------
    Tensor
        The Riemann zeta function evaluated at each element of `input`.
    """
    return torchscience.ops.torchscience._riemann_zeta(input)
