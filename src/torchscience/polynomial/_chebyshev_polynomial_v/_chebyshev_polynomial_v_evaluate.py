import warnings

from torch import Tensor

from torchscience.polynomial._chebyshev_polynomial_v._chebyshev_polynomial_v import (
    ChebyshevPolynomialV,
)


def chebyshev_polynomial_v_evaluate(
    c: ChebyshevPolynomialV,
    x: Tensor,
) -> Tensor:
    """Evaluate Chebyshev V series at points using Clenshaw's algorithm.

    Parameters
    ----------
    c : ChebyshevPolynomialV
        Chebyshev V series with coefficients shape (...batch, N).
    x : Tensor
        Evaluation points, shape (...x_batch).

    Returns
    -------
    Tensor
        Values c(x), shape is broadcast of c's batch dims with x's shape.

    Warnings
    --------
    UserWarning
        If any evaluation points are outside the natural domain [-1, 1].

    Notes
    -----
    Uses Clenshaw's algorithm adapted for Chebyshev V polynomials.

    The Chebyshev V polynomials satisfy the recurrence:
        V_0(x) = 1
        V_1(x) = 2x - 1
        V_{n+1}(x) = 2x * V_n(x) - V_{n-1}(x)

    The Clenshaw algorithm computes:
        b_{n+1} = b_{n+2} = 0
        b_k = c_k + 2*x*b_{k+1} - b_{k+2}  for k = n, n-1, ..., 1
        f(x) = c_0 + (2*x - 1)*b_1 - b_2

    Examples
    --------
    >>> c = chebyshev_polynomial_v(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2*V_1 + 3*V_2
    >>> chebyshev_polynomial_v_evaluate(c, torch.tensor([0.0]))
    """
    # Domain check only applies to real tensors (complex roots are expected)
    if not x.is_complex():
        domain = ChebyshevPolynomialV.DOMAIN

        if ((x < domain[0]) | (x > domain[1])).any():
            warnings.warn(
                f"Evaluating ChebyshevPolynomialV outside natural domain "
                f"[{domain[0]}, {domain[1]}]. Results may be numerically unstable.",
                stacklevel=2,
            )

    coeffs = c.coeffs
    n = coeffs.shape[-1]

    # Handle trivial cases
    if n == 0:
        return x * 0.0
    if n == 1:
        return (
            coeffs[..., 0].expand_as(x).clone()
            if coeffs.dim() == 1
            else coeffs[..., 0:1] * (x * 0.0 + 1.0)
        )

    # Non-batched case (1D coeffs)
    if coeffs.dim() == 1:
        b_kp2 = x * 0.0  # b_{n+2} = 0
        b_kp1 = x * 0.0  # b_{n+1} = 0

        for k in range(n - 1, 0, -1):
            b_k = coeffs[k] + 2.0 * x * b_kp1 - b_kp2
            b_kp2 = b_kp1
            b_kp1 = b_k

        # Final step for V polynomials: f(x) = c_0 + (2*x - 1)*b_1 - b_2
        # This accounts for V_1(x) = 2x - 1 and V_0(x) = 1
        return coeffs[0] + (2.0 * x - 1.0) * b_kp1 - b_kp2

    # Batched case: coeffs shape (...batch, N), x shape (...x_batch)
    batch_shape = coeffs.shape[:-1]

    # Expand coeffs for broadcasting: (...batch, 1, 1, ..., N)
    coeffs_expanded = coeffs
    for _ in range(x.dim()):
        coeffs_expanded = coeffs_expanded.unsqueeze(-2)

    # Expand x for broadcasting: (1, 1, ..., ...x_batch)
    x_expanded = x
    for _ in range(len(batch_shape)):
        x_expanded = x_expanded.unsqueeze(0)

    b_kp2 = x_expanded * 0.0
    b_kp1 = x_expanded * 0.0

    for k in range(n - 1, 0, -1):
        b_k = coeffs_expanded[..., k] + 2.0 * x_expanded * b_kp1 - b_kp2
        b_kp2 = b_kp1
        b_kp1 = b_k

    return coeffs_expanded[..., 0] + (2.0 * x_expanded - 1.0) * b_kp1 - b_kp2
