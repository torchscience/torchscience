from torch import Tensor

import torchscience.ops.torchscience

__all__ = ["polygamma"]


def polygamma(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Polygamma function :math:`\psi^{(n)}(x)`.

    The polygamma function is the :math:`n`-th derivative of the digamma function:

    .. math::
        \psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x) = \frac{d^{n+1}}{dx^{n+1}} \ln \Gamma(x)

    Special cases:

    - :math:`\psi^{(0)}(x) = \psi(x)` is the digamma function
    - :math:`\psi^{(1)}(x)` is the trigamma function
    - :math:`\psi^{(2)}(x)` is the tetragamma function

    Parameters
    ----------
    n : Tensor
        Non-negative integer order (:math:`n \geq 0`).
    x : Tensor
        Argument (real, :math:`x > 0` for positive integer :math:`n`).

    Returns
    -------
    Tensor
        The polygamma function :math:`\psi^{(n)}(x)`.
    """
    return torchscience.ops.torchscience._polygamma(n, x)
