from typing import Optional, Union

import torch
from torch import Tensor

from ._ultraspherical_window import _gegenbauer_polynomial


def periodic_ultraspherical_window(
    n: int,
    mu: Union[float, Tensor],
    x_mu: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Ultraspherical (Gegenbauer) window function (periodic).

    Computes a periodic ultraspherical window of length n. The periodic version
    is designed for spectral analysis where the window will be used with DFT/FFT.

    Mathematical Definition
    -----------------------
    The periodic ultraspherical window uses a denominator of n (instead of n-1)
    for proper periodicity. The window is defined via its frequency response:

        W(theta) = C_{N-1}^{mu}(x_mu * cos(theta)) / C_{N-1}^{mu}(x_mu)

    where C_n^{mu}(x) is the Gegenbauer polynomial of degree n with parameter mu.
    The time-domain window is obtained by inverse Fourier transform.

    Properties
    ----------
    - Generalizes the Dolph-Chebyshev window (mu -> 0 limit)
    - mu controls the trade-off between main lobe width and sidelobe decay
    - x_mu > 1 controls the sidelobe level
    - Designed for spectral analysis with FFT

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    mu : float or Tensor
        Gegenbauer polynomial parameter (mu > 0).
        Controls the sidelobe roll-off characteristics.
        - mu near 0: approaches Dolph-Chebyshev (equiripple sidelobes)
        - mu = 1: Saramaki window
        - mu = 2: faster sidelobe decay
    x_mu : float or Tensor
        Sidelobe level parameter (x_mu > 1).
        Controls the ratio of main lobe to sidelobe amplitude.
        Larger values give lower sidelobes but wider main lobe.
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
    This function supports autograd - gradients flow through mu and x_mu.

    References
    ----------
    .. [1] S. W. A. Bergen and A. Antoniou, "Design of Ultraspherical Window
           Functions with Prescribed Spectral Characteristics," EURASIP Journal
           on Applied Signal Processing, vol. 2004, no. 13, pp. 2053-2065, 2004.

    See Also
    --------
    ultraspherical_window : Symmetric version.
    periodic_dolph_chebyshev_window : Special case when mu -> 0.
    """
    if n < 0:
        raise ValueError(
            f"periodic_ultraspherical_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    # Convert parameters to tensors
    target_dtype = dtype or torch.float32

    if not isinstance(mu, Tensor):
        mu = torch.tensor(mu, dtype=target_dtype, device=device)
    if not isinstance(x_mu, Tensor):
        x_mu = torch.tensor(x_mu, dtype=target_dtype, device=device)

    # Ensure consistent dtype
    if mu.dtype != target_dtype:
        mu = mu.to(dtype=target_dtype)
    if x_mu.dtype != target_dtype:
        x_mu = x_mu.to(dtype=target_dtype)

    # Validate parameters
    if mu <= 0:
        raise ValueError(
            f"periodic_ultraspherical_window: mu must be positive, got {mu}"
        )
    if x_mu <= 1:
        raise ValueError(
            f"periodic_ultraspherical_window: x_mu must be > 1, got {x_mu}"
        )

    target_device = device or mu.device

    # For a periodic N-point window, we use a Fourier series with denominator n
    # instead of n-1 to ensure proper periodicity for FFT applications.

    # Sample indices for window output
    k = torch.arange(n, dtype=target_dtype, device=target_device)

    # Compute window using direct Fourier series summation
    # For a periodic window, center is at n/2
    center = n / 2.0
    window = torch.zeros(n, dtype=target_dtype, device=target_device)

    # Number of frequency components to sum
    n_freqs = n // 2 + 1

    for m in range(n_freqs):
        # Frequency (normalized angular frequency) - use n as denominator for periodic
        omega = torch.pi * m / n if n > 0 else torch.tensor(0.0)

        # Argument to Gegenbauer polynomial: x_mu * cos(omega)
        arg = x_mu * torch.cos(
            torch.tensor(omega, dtype=target_dtype, device=target_device)
        )

        # Evaluate frequency response: C_{N-1}^{mu}(arg) / C_{N-1}^{mu}(x_mu)
        c_n_arg = _gegenbauer_polynomial(n - 1, mu, arg.unsqueeze(0))
        c_n_x_mu = _gegenbauer_polynomial(n - 1, mu, x_mu.unsqueeze(0))
        freq_mag = c_n_arg / c_n_x_mu

        # Compute cosine term for each output sample
        cosine_term = torch.cos(omega * (k - center))

        # Weighting: DC and Nyquist (if present) are weighted by 1, others by 2
        weight = 1.0 if (m == 0 or (n % 2 == 0 and m == n // 2)) else 2.0

        window = window + weight * freq_mag * cosine_term

    # Normalize so maximum is 1
    window = window / window.abs().max()

    if dtype is not None and window.dtype != dtype:
        window = window.to(dtype=dtype)

    return window
