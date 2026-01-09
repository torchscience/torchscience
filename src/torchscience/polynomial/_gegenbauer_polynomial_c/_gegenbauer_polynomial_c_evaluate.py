import warnings

from torch import Tensor

from torchscience.polynomial._gegenbauer_polynomial_c._gegenbauer_polynomial_c import (
    GegenbauerPolynomialC,
)


def gegenbauer_polynomial_c_evaluate(
    c: GegenbauerPolynomialC,
    x: Tensor,
) -> Tensor:
    """Evaluate Gegenbauer series at points using Clenshaw's algorithm.

    Parameters
    ----------
    c : GegenbauerPolynomialC
        Gegenbauer series with coefficients shape (...batch, N) and parameter lambda_.
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

    The Gegenbauer polynomials satisfy the recurrence:
        C_0^{lambda}(x) = 1
        C_1^{lambda}(x) = 2*lambda*x
        C_{k+1}^{lambda}(x) = (2*(k+lambda)/(k+1)) * x * C_k^{lambda}(x)
                           - ((k+2*lambda-1)/(k+1)) * C_{k-1}^{lambda}(x)

    In standard form: C_{k+1}(x) = A_k * x * C_k(x) - C_k' * C_{k-1}(x)
    where A_k = 2*(k+lambda)/(k+1) and C_k' = (k+2*lambda-1)/(k+1)

    For the Clenshaw backward recurrence to evaluate f(x) = sum(c_k * C_k^{lambda}(x)):
        b_{n+1} = b_{n+2} = 0
        b_k = c_k + A_k * x * b_{k+1} - C_{k+1}' * b_{k+2}  for k = n-1, ..., 1, 0
        f(x) = b_0

    Examples
    --------
    >>> c = gegenbauer_polynomial_c(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(1.0))
    >>> gegenbauer_polynomial_c_evaluate(c, torch.tensor([0.0]))
    tensor([-2.])  # 1 + 0 + 3*(-1) = -2 for C_2^1(0) = -1
    """
    # Domain check only applies to real tensors (complex roots are expected)
    if not x.is_complex():
        domain = GegenbauerPolynomialC.DOMAIN

        if ((x < domain[0]) | (x > domain[1])).any():
            warnings.warn(
                f"Evaluating GegenbauerPolynomialC outside natural domain "
                f"[{domain[0]}, {domain[1]}]. Results may be numerically unstable.",
                stacklevel=2,
            )

    coeffs = c.coeffs
    lambda_ = c.lambda_
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

        # Clenshaw backward recurrence for Gegenbauer
        # b_k = c_k + A_k * x * b_{k+1} - C_{k+1}' * b_{k+2}
        # where A_k = 2*(k+lambda)/(k+1) and C_k' = (k+2*lambda-1)/(k+1)
        for k in range(n - 2, -1, -1):
            # A_k = 2*(k+lambda)/(k+1)
            a_k = 2.0 * (k + lambda_) / (k + 1.0)
            # C_{k+1}' = ((k+1)+2*lambda-1)/(k+2) = (k+2*lambda)/(k+2)
            c_kp1 = (k + 2.0 * lambda_) / (k + 2.0)

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

    # Expand lambda_ for broadcasting
    lambda_expanded = lambda_
    if lambda_.dim() == 0:
        pass  # Scalar, will broadcast automatically
    else:
        for _ in range(x.dim()):
            lambda_expanded = lambda_expanded.unsqueeze(-1)

    b_kp2 = x_expanded * 0.0
    b_kp1 = x_expanded * 0.0 + coeffs_expanded[..., n - 1]

    for k in range(n - 2, -1, -1):
        # A_k = 2*(k+lambda)/(k+1)
        a_k = 2.0 * (k + lambda_expanded) / (k + 1.0)
        # C_{k+1}' = (k+2*lambda)/(k+2)
        c_kp1 = (k + 2.0 * lambda_expanded) / (k + 2.0)

        b_k = (
            coeffs_expanded[..., k] + a_k * x_expanded * b_kp1 - c_kp1 * b_kp2
        )
        b_kp2 = b_kp1
        b_kp1 = b_k

    return b_kp1
