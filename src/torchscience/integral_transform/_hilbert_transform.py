"""Hilbert transform implementation."""

from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def hilbert_transform(
    input: Tensor,
    *,
    n: Optional[int] = None,
    dim: int = -1,
) -> Tensor:
    r"""Compute the Hilbert transform of a signal along a specified dimension.

    The Hilbert transform is defined as:

    .. math::
        \mathcal{H}[f](x) = \frac{1}{\pi} \text{PV} \int_{-\infty}^{\infty}
        \frac{f(t)}{t - x} \, dt

    where PV denotes the Cauchy principal value.

    For discrete signals, this is computed efficiently using the FFT by
    multiplying the frequency spectrum by :math:`-i \cdot \text{sign}(\omega)`.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. Can be real or complex.
    n : int, optional
        Signal length. If given, the input will either be zero-padded or
        truncated to this length before computing the transform. This is
        useful for mitigating edge effects by zero-padding.
        Default: ``None`` (use input size along ``dim``).
    dim : int, optional
        The dimension along which to compute the transform.
        Default: ``-1`` (last dimension).

    Returns
    -------
    Tensor
        The Hilbert transform of the input.
        If ``n`` is specified and differs from the input size along ``dim``,
        the output size along ``dim`` will be ``n``.
        If input is real, output is real (the imaginary part is discarded).

    Examples
    --------
    Basic usage with a sine wave:

    >>> t = torch.linspace(0, 2 * torch.pi, 100)
    >>> x = torch.sin(t)  # sin(t)
    >>> h = torchscience.integral_transform.hilbert_transform(x)
    >>> # H[sin(t)] ≈ -cos(t) for positive frequencies
    >>> torch.allclose(h, -torch.cos(t), atol=0.1)
    True

    Transform along a specific dimension:

    >>> x = torch.randn(3, 100)
    >>> h = torchscience.integral_transform.hilbert_transform(x, dim=1)
    >>> h.shape
    torch.Size([3, 100])

    Zero-padding to mitigate edge effects:

    >>> x = torch.randn(100)
    >>> h = torchscience.integral_transform.hilbert_transform(x, n=256)
    >>> h.shape  # Output has length 256
    torch.Size([256])

    Notes
    -----
    **Mathematical Properties:**

    - :math:`\mathcal{H}[\sin(\omega t)] = -\cos(\omega t)` (for positive :math:`\omega`)
    - :math:`\mathcal{H}[\cos(\omega t)] = \sin(\omega t)` (for positive :math:`\omega`)
    - :math:`\mathcal{H}[\mathcal{H}[f]] = -f` (involutory up to sign)
    - Energy preservation: :math:`\int |H[f]|^2 = \int |f|^2`
    - Linearity: :math:`\mathcal{H}[\alpha f + \beta g] = \alpha\mathcal{H}[f] + \beta\mathcal{H}[g]`

    **Complex Input Behavior:**

    For complex inputs, the transform is applied linearly to both components:
    :math:`\mathcal{H}[a + ib] = \mathcal{H}[a] + i\mathcal{H}[b]`. This
    preserves conjugate symmetry: :math:`\mathcal{H}[\overline{f}] = \overline{\mathcal{H}[f]}`.

    **Analytic Signal:**

    The analytic signal is defined as :math:`z(t) = f(t) + i\mathcal{H}[f](t)`.
    It can be computed as:

    >>> analytic = x + 1j * hilbert_transform(x)

    **Implementation:**

    Uses FFT-based computation:

    1. Compute FFT of input (with optional zero-padding via ``n``)
    2. Multiply by frequency response :math:`h[k] = -i \cdot \text{sign}(\text{freq}[k])`
    3. Compute inverse FFT

    **Gradient Computation:**

    Gradients are computed analytically using the property that the Hilbert
    transform is anti-self-adjoint: :math:`\mathcal{H}^T = -\mathcal{H}`.
    Therefore, for a loss :math:`L`:

    .. math::
        \frac{\partial L}{\partial x} = -\mathcal{H}\left[\frac{\partial L}{\partial y}\right]

    where :math:`y = \mathcal{H}[x]`. Second-order gradients are also supported.

    Warnings
    --------
    - Edge effects: The discrete Hilbert transform assumes periodic boundary
      conditions. For non-periodic signals, use zero-padding via the ``n``
      parameter (e.g., ``n=2*input.size(dim)``).

    - The transform is not well-defined for DC components (frequency = 0).

    References
    ----------
    .. [1] F.W. King, "Hilbert Transforms," Cambridge University Press, 2009.

    .. [2] S.L. Hahn, "Hilbert Transforms in Signal Processing,"
           Artech House, 1996.

    See Also
    --------
    inverse_hilbert_transform : The inverse Hilbert transform.
    scipy.signal.hilbert : SciPy's Hilbert transform (returns analytic signal).
    """
    return torch.ops.torchscience.hilbert_transform(
        input, n if n is not None else -1, dim
    )
