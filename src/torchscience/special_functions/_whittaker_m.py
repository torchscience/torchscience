from torch import Tensor

import torchscience.ops.torchscience


def whittaker_m(kappa: Tensor, mu: Tensor, z: Tensor) -> Tensor:
    r"""
    Whittaker M function :math:`M_{\kappa,\mu}(z)`.

    The Whittaker M function is a solution to Whittaker's differential equation:

    .. math::

        \frac{d^2 W}{dz^2} + \left(-\frac{1}{4} + \frac{\kappa}{z} + \frac{1/4 - \mu^2}{z^2}\right) W = 0

    It is related to the confluent hypergeometric function by:

    .. math::

        M_{\kappa,\mu}(z) = z^{\mu + 1/2} e^{-z/2} {}_1F_1\left(\mu - \kappa + \frac{1}{2}; 2\mu + 1; z\right)

    where :math:`{}_1F_1(a; b; z)` is the confluent hypergeometric function of the first kind.

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
        Value of the Whittaker M function :math:`M_{\kappa,\mu}(z)`.
    """
    return torchscience.ops.torchscience._whittaker_m(kappa, mu, z)
