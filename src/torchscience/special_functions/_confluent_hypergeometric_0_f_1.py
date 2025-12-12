from torch import Tensor

import torch


def confluent_hypergeometric_0_f_1(b: Tensor, z: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Confluent hypergeometric limit function.

    .. math::
        {}_0F_1(; b; z) = \sum_{k=0}^{\infty} \frac{z^k}{(b)_k k!}

    where :math:`(b)_k` is the Pochhammer symbol (rising factorial).

    This function is related to Bessel functions:

    .. math::
        J_\nu(z) = \frac{(z/2)^\nu}{\Gamma(\nu+1)} {}_0F_1(; \nu+1; -z^2/4)

    Parameters
    ----------
    b : Tensor
        Parameter of the hypergeometric function.
    z : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Confluent hypergeometric limit function.
    """
    output = torch.ops.torchscience._confluent_hypergeometric_0_f_1(b, z)

    if out is not None:
        out.copy_(output)
        return out

    return output
