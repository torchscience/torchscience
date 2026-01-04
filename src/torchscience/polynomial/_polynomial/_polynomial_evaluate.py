from __future__ import annotations

from torch import Tensor

from torchscience.polynomial import Polynomial


def polynomial_evaluate(p: Polynomial, x: Tensor) -> Tensor:
    """Evaluate polynomial at points using Horner's method.

    Parameters
    ----------
    p : Polynomial
        Polynomial with coefficients shape (...batch, N).
    x : Tensor
        Evaluation points, shape (...x_batch). The result shape is
        (...batch, ...x_batch) where batch dimensions of p broadcast
        with x.

    Returns
    -------
    Tensor
        Values p(x), shape is broadcast of p's batch dims with x's shape.

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> polynomial_evaluate(p, torch.tensor([0.0, 1.0, 2.0]))
    tensor([ 1.,  6., 17.])
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    # For non-batched polynomial (1D coeffs), just do standard Horner
    if coeffs.dim() == 1:
        result = coeffs[n - 1].expand_as(x).clone()
        for i in range(n - 2, -1, -1):
            result = result * x + coeffs[i]
        return result

    # For batched polynomial, we need to handle the broadcasting more carefully
    # coeffs shape: (...batch, N)
    # x shape: (...x_batch)
    # result shape: (...batch, ...x_batch)

    batch_shape = coeffs.shape[:-1]  # (...batch)
    x_shape = x.shape  # (...x_batch)

    # Reshape coeffs to (...batch, 1, 1, ..., N) to broadcast with x
    # Add x.dim() singleton dimensions before the coefficient dimension
    coeffs_expanded = coeffs
    for _ in range(x.dim()):
        # Insert dimension before the last (coefficient) dimension
        coeffs_expanded = coeffs_expanded.unsqueeze(-2)
    # Now coeffs_expanded has shape (...batch, 1, 1, ..., N)

    # Reshape x to (1, 1, ..., ...x_batch) to broadcast with batch dims
    x_expanded = x
    for _ in range(len(batch_shape)):
        x_expanded = x_expanded.unsqueeze(0)
    # Now x_expanded has shape (1, 1, ..., ...x_batch)

    # Start Horner's method
    result = coeffs_expanded[
        ..., n - 1
    ]  # shape: (...batch, 1, 1, ...) or broadcast result

    for i in range(n - 2, -1, -1):
        result = result * x_expanded + coeffs_expanded[..., i]

    return result
