import warnings

from torch import Tensor

from torchscience.polynomial._legendre_polynomial_p._legendre_polynomial_p import (
    LegendrePolynomialP,
)


def legendre_polynomial_p_evaluate(
    c: LegendrePolynomialP,
    x: Tensor,
) -> Tensor:
    """Evaluate Legendre series at points using Clenshaw's algorithm.

    Parameters
    ----------
    c : LegendrePolynomialP
        Legendre series with coefficients shape (...batch, N).
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
    Uses Clenshaw's algorithm for numerical stability.

    The Legendre polynomials satisfy the recurrence:
        P_0(x) = 1
        P_1(x) = x
        P_{k+1}(x) = ((2k+1)/(k+1)) * x * P_k(x) - (k/(k+1)) * P_{k-1}(x)

    In standard form: P_{k+1}(x) = A_k * x * P_k(x) - C_k * P_{k-1}(x)
    where A_k = (2k+1)/(k+1) and C_k = k/(k+1)

    For the Clenshaw backward recurrence to evaluate f(x) = sum(c_k * P_k(x)):
        b_{n+1} = b_{n+2} = 0
        b_k = c_k + A_k * x * b_{k+1} - C_{k+1} * b_{k+2}  for k = n-1, ..., 1, 0
        f(x) = b_0

    Examples
    --------
    >>> c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))  # 1*P_0 + 2*P_1 + 3*P_2
    >>> legendre_polynomial_p_evaluate(c, torch.tensor([0.0]))
    tensor([-0.5])  # 1 + 0 + 3*(-1/2) = -0.5
    """
    # Domain check only applies to real tensors (complex roots are expected)
    if not x.is_complex():
        domain = LegendrePolynomialP.DOMAIN

        if ((x < domain[0]) | (x > domain[1])).any():
            warnings.warn(
                f"Evaluating LegendrePolynomialP outside natural domain "
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
        b_kp1 = x * 0.0 + coeffs[n - 1]  # b_{n-1} = c_{n-1}

        # Clenshaw backward recurrence for Legendre
        # b_k = c_k + A_k * x * b_{k+1} - C_{k+1} * b_{k+2}
        # where A_k = (2k+1)/(k+1), C_k = k/(k+1)
        for k in range(n - 2, -1, -1):
            # A_k = (2k+1)/(k+1)
            a_k = (2.0 * k + 1.0) / (k + 1.0)
            # C_{k+1} = (k+1)/(k+2)
            c_kp1 = (k + 1.0) / (k + 2.0)

            b_k = coeffs[k] + a_k * x * b_kp1 - c_kp1 * b_kp2
            b_kp2 = b_kp1
            b_kp1 = b_k

        return b_kp1

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
    b_kp1 = x_expanded * 0.0 + coeffs_expanded[..., n - 1]

    for k in range(n - 2, -1, -1):
        # A_k = (2k+1)/(k+1)
        a_k = (2.0 * k + 1.0) / (k + 1.0)
        # C_{k+1} = (k+1)/(k+2)
        c_kp1 = (k + 1.0) / (k + 2.0)

        b_k = (
            coeffs_expanded[..., k] + a_k * x_expanded * b_kp1 - c_kp1 * b_kp2
        )
        b_kp2 = b_kp1
        b_kp1 = b_k

    return b_kp1
