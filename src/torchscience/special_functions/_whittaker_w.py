from torch import Tensor

import torchscience.ops.torchscience


def whittaker_w(kappa: Tensor, mu: Tensor, z: Tensor) -> Tensor:
    r"""
    Whittaker W function :math:`W_{\kappa,\mu}(z)`.

    The Whittaker W function is a solution to Whittaker's differential equation:

    .. math::

        \frac{d^2 W}{dz^2} + \left(-\frac{1}{4} + \frac{\kappa}{z} + \frac{1/4 - \mu^2}{z^2}\right) W = 0

    It is related to the Tricomi confluent hypergeometric function by:

    .. math::

        W_{\kappa,\mu}(z) = z^{\mu + 1/2} e^{-z/2} U\left(\mu - \kappa + \frac{1}{2}, 2\mu + 1, z\right)

    where :math:`U(a, b, z)` is Tricomi's confluent hypergeometric function.

    Parameters
    ----------
    kappa : Tensor
        First parameter :math:`\kappa`.
    mu : Tensor
        Second parameter :math:`\mu`.
    z : Tensor
        Argument.

    Returns
    -------
    Tensor
        Value of the Whittaker W function :math:`W_{\kappa,\mu}(z)`.
    """
    return torchscience.ops.torchscience._whittaker_w(kappa, mu, z)
