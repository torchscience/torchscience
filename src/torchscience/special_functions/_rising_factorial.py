import torch
from torch import Tensor


def rising_factorial(x: Tensor, n: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Computes the rising factorial (Pochhammer symbol) of x with n terms.

    .. math::

        \\text{rising\\_factorial}(x, n) = (x)_n = x \\cdot (x+1) \\cdot (x+2) \\cdots (x+n-1) = \\frac{\\Gamma(x+n)}{\\Gamma(x)}

    Also known as the Pochhammer symbol or rising sequential product.

    Parameters
    ----------
    x : Tensor
        Base value.

    n : Tensor
        Number of terms (non-negative integer).

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The rising factorial of x with n terms.

    Examples
    --------
    >>> rising_factorial(torch.tensor([2.0, 3.0, 4.0]), torch.tensor([3.0, 3.0, 3.0]))
    tensor([24., 60., 120.])

    >>> # (2)_3 = 2 * 3 * 4 = 24
    >>> # (3)_3 = 3 * 4 * 5 = 60
    >>> # (4)_3 = 4 * 5 * 6 = 120
    """
    output: Tensor = torch.ops.torchscience._rising_factorial(x, n)

    if out is not None:
        out.copy_(output)

        return out

    return output
