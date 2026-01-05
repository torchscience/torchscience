"""ChebyshevT tensorclass for Chebyshev series of the first kind."""

from __future__ import annotations

from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._polynomial_error import PolynomialError


@tensorclass
class ChebyshevT:
    """Chebyshev series of the first kind.

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * T_k(x)

    where T_k(x) are Chebyshev polynomials of the first kind.

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., k] is the coefficient of T_k(x).
        Batch dimensions come first, coefficient dimension last.

    Notes
    -----
    The standard domain for Chebyshev polynomials is [-1, 1].
    """

    coeffs: Tensor

    def __call__(self, x: Tensor) -> Tensor:
        from ._chebyshev_t_evaluate import chebyshev_t_evaluate

        return chebyshev_t_evaluate(self, x)

    def __add__(self, other: "ChebyshevT") -> "ChebyshevT":
        from ._chebyshev_t_add import chebyshev_t_add

        return chebyshev_t_add(self, other)

    def __radd__(self, other: "ChebyshevT") -> "ChebyshevT":
        from ._chebyshev_t_add import chebyshev_t_add

        return chebyshev_t_add(other, self)

    def __sub__(self, other: "ChebyshevT") -> "ChebyshevT":
        from ._chebyshev_t_subtract import chebyshev_t_subtract

        return chebyshev_t_subtract(self, other)

    def __rsub__(self, other: "ChebyshevT") -> "ChebyshevT":
        from ._chebyshev_t_subtract import chebyshev_t_subtract

        return chebyshev_t_subtract(other, self)

    def __neg__(self) -> "ChebyshevT":
        from ._chebyshev_t_negate import chebyshev_t_negate

        return chebyshev_t_negate(self)

    def __mul__(self, other):
        if isinstance(other, ChebyshevT):
            from ._chebyshev_t_multiply import chebyshev_t_multiply

            return chebyshev_t_multiply(self, other)
        if isinstance(other, Tensor):
            from ._chebyshev_t_scale import chebyshev_t_scale

            return chebyshev_t_scale(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, ChebyshevT):
            from ._chebyshev_t_multiply import chebyshev_t_multiply

            return chebyshev_t_multiply(other, self)
        if isinstance(other, Tensor):
            from ._chebyshev_t_scale import chebyshev_t_scale

            return chebyshev_t_scale(self, other)
        return NotImplemented

    def __pow__(self, n: int) -> "ChebyshevT":
        from ._chebyshev_t_pow import chebyshev_t_pow

        return chebyshev_t_pow(self, n)


def chebyshev_t(coeffs: Tensor) -> ChebyshevT:
    """Create Chebyshev series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of T_k(x).
        Must have at least one coefficient.

    Returns
    -------
    ChebyshevT
        Chebyshev series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))  # 1*T_0 + 2*T_1 + 3*T_2
    >>> c.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Chebyshev series must have at least one coefficient"
        )

    return ChebyshevT(coeffs=coeffs)
