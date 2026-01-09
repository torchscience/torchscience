from typing import Optional, Union

import torch
from torch import Tensor


def periodic_confined_gaussian_window(
    n: int,
    std: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Confined Gaussian window function (periodic).

    Computes a periodic confined Gaussian window of length n. The confined
    Gaussian window is a modification of the Gaussian window that is designed
    to reach exactly zero at the boundaries, unlike a standard truncated
    Gaussian.

    Mathematical Definition
    -----------------------
    The periodic confined Gaussian window is defined as:

        w[k] = g(x_k) - g(L/2) * [g(x_k - L) + g(x_k + L)] / [g(-3L/2) + g(L/2)]

    where:
        - g(x) = exp(-x^2 / (2 * sigma^2)) is the Gaussian function
        - x_k = k - L/2 is the centered sample position
        - L = n is the window length (periodic denominator)
        - sigma = std * L is the standard deviation

    for k = 0, 1, ..., n-1.

    The correction term (second part of the formula) ensures that the window
    reaches exactly zero at the boundaries while maintaining the Gaussian
    shape near the center.

    Properties
    ----------
    - Reaches exactly zero at k = 0
    - Smooth bell-shaped curve like Gaussian
    - Better defined finite support than truncated Gaussian
    - Useful when strict zero boundary conditions are required

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    std : float or Tensor
        Standard deviation parameter controlling the window width, specified
        relative to the window length. The actual standard deviation is
        sigma = std * n. Typical values are between 0.1 and 0.5.
        Larger values produce wider windows.
    dtype : torch.dtype, optional
        The desired data type of the returned tensor.
    layout : torch.layout, optional
        The desired layout of the returned tensor.
    device : torch.device, optional
        The desired device of the returned tensor.

    Returns
    -------
    Tensor
        A 1-D tensor of size (n,) containing the window values.

    Notes
    -----
    This function supports autograd - gradients flow through the std parameter.

    See Also
    --------
    confined_gaussian_window : Symmetric version for filter design.
    periodic_gaussian_window : Standard periodic Gaussian (without confinement).
    """
    if n < 0:
        raise ValueError(
            f"periodic_confined_gaussian_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    if not isinstance(std, Tensor):
        target_dtype = dtype or torch.float32
        std = torch.tensor(std, dtype=target_dtype, device=device)

    # Window length (periodic: use n as denominator)
    L = float(n)
    half_L = L / 2.0

    # Actual standard deviation
    sigma = std * L

    # Sample positions centered at L/2
    k = torch.arange(n, dtype=std.dtype, device=std.device)
    x = k - half_L  # x_k = k - L/2

    # Define Gaussian function: g(t) = exp(-t^2 / (2 * sigma^2))
    def g(t: Tensor) -> Tensor:
        return torch.exp(-t * t / (2.0 * sigma * sigma))

    # Main Gaussian term
    g_x = g(x)

    # Correction term to force zeros at boundaries
    # g(L/2) * [g(x - L) + g(x + L)] / [g(-3L/2) + g(L/2)]
    g_half_L = g(torch.tensor(half_L, dtype=std.dtype, device=std.device))
    g_neg_3half_L = g(
        torch.tensor(-1.5 * L, dtype=std.dtype, device=std.device)
    )

    numerator = g(x - L) + g(x + L)
    denominator = g_neg_3half_L + g_half_L

    correction = g_half_L * numerator / denominator

    # Confined Gaussian window
    window = g_x - correction

    if dtype is not None and window.dtype != dtype:
        window = window.to(dtype=dtype)

    return window
