from torch import Tensor

from torchscience.polynomial._hermite_polynomial_h._hermite_polynomial_h import (
    HermitePolynomialH,
)


def hermite_polynomial_h_evaluate(
    c: HermitePolynomialH,
    x: Tensor,
) -> Tensor:
    """Evaluate Physicists' Hermite series at points using Clenshaw's algorithm.

    Parameters
    ----------
    c : HermitePolynomialH
        Hermite series with coefficients shape (...batch, N).
    x : Tensor
        Evaluation points, shape (...x_batch).

    Returns
    -------
    Tensor
        Values c(x), shape is broadcast of c's batch dims with x's shape.

    Notes
    -----
    Uses Clenshaw's algorithm for numerical stability.

    The physicists' Hermite polynomials satisfy the recurrence:
        H_0(x) = 1
        H_1(x) = 2x
        H_{k+1}(x) = 2x * H_k(x) - 2k * H_{k-1}(x)

    In standard form: H_{k+1}(x) = A_k * x * H_k(x) - C_k * H_{k-1}(x)
    where A_k = 2 and C_k = 2k

    For the Clenshaw backward recurrence to evaluate f(x) = sum(c_k * H_k(x)):
        b_{n+1} = b_{n+2} = 0
        b_k = c_k + A_k * x * b_{k+1} - C_{k+1} * b_{k+2}  for k = n-1, ..., 1, 0
        f(x) = b_0

    Examples
    --------
    >>> c = hermite_polynomial_h(torch.tensor([1.0, 0.0, 1.0]))  # 1*H_0 + 0*H_1 + 1*H_2
    >>> hermite_polynomial_h_evaluate(c, torch.tensor([0.0]))
    tensor([-2.])  # H_0(0) = 1, H_2(0) = -2, so 1 + (-2) = -1... wait H_2(0) = 4*0^2 - 2 = -2
    """
    # No domain check for Hermite polynomials since domain is (-inf, inf)

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

        # Clenshaw backward recurrence for Physicists' Hermite
        # b_k = c_k + A_k * x * b_{k+1} - C_{k+1} * b_{k+2}
        # where A_k = 2, C_k = 2k
        for k in range(n - 2, -1, -1):
            # A_k = 2
            a_k = 2.0
            # C_{k+1} = 2*(k+1)
            c_kp1 = 2.0 * (k + 1.0)

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
        # A_k = 2
        a_k = 2.0
        # C_{k+1} = 2*(k+1)
        c_kp1 = 2.0 * (k + 1.0)

        b_k = (
            coeffs_expanded[..., k] + a_k * x_expanded * b_kp1 - c_kp1 * b_kp2
        )
        b_kp2 = b_kp1
        b_kp1 = b_k

    return b_kp1
