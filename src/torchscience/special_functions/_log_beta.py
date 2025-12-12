import torch
from torch import Tensor


def log_beta(a: Tensor, b: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Logarithm of the beta function.

    .. math::

        \\log B(a, b) = \\log\\Gamma(a) + \\log\\Gamma(b) - \\log\\Gamma(a + b)

    where :math:`\\Gamma` is the gamma function and :math:`B` is the beta function.

    This is more numerically stable than computing ``log(beta(a, b))`` directly
    for large values of ``a`` and ``b``.

    Parameters
    ----------
    a : Tensor, shape=(...)
        First input tensor.

    b : Tensor, shape=(...)
        Second input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The logarithm of the beta function evaluated at each element pair.

    Examples
    --------
    >>> log_beta(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
    tensor([ 0.0000, -1.7918, -3.4012])

    >>> log_beta(torch.tensor([0.5]), torch.tensor([0.5]))
    tensor([1.1447])
    """
    output: Tensor = torch.ops.torchscience._log_beta(a, b)

    if out is not None:
        out.copy_(output)

        return out

    return output
