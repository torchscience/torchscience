import torch
from torch import Tensor

from ._common import dispatch_colored_noise


def violet_noise(
    size: int,
    *,
    generator=None,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
) -> Tensor:
    r"""
    Violet (purple) noise.

    Generates violet noise of the given size. Violet noise has a power
    spectral density proportional to the square of frequency:

    .. math::
        S(f) \propto f^2.

    It is the spectral mirror of brown noise: heavy emphasis on high
    frequencies, with power per octave growing eightfold per octave shift.
    Equivalently, violet noise is the discrete-time derivative of white
    noise -- differentiation in time multiplies the spectrum by ``f``, which
    multiplies the PSD by ``f^2``.

    Algorithm
    ---------

    1. Draw N i.i.d. samples from N(0, 1).
    2. Take the real FFT.
    3. Multiply each bin by ``|f|``, giving |X(f)| ~ f and hence
       |X(f)|^2 ~ f^2.
    4. The DC bin is naturally zero because the filter ``f`` vanishes at
       f=0; the time-domain output is therefore exactly zero-mean.
    5. Inverse FFT, then normalize by max absolute value so peak amplitude
       is 1.

    Testable Guarantees
    -------------------
    - PSD log-log slope equals +2 (slope test).
    - Power x8 per octave (octave-power-ratio test).
    - Mean is zero up to FFT round-off.
    - Peak absolute value is exactly 1.

    Applications
    ------------
    - Differential-privacy mechanisms and emphasis filters that need
      strong high-frequency content.
    - Models for noise sources whose mechanism is itself a derivative
      (e.g. velocity-noise driving a position).

    Dtype Support
    -------------
    - Supports float16, bfloat16, float32, float64.

    Parameters
    ----------
    size : int
        Length of the output 1-D tensor.
    generator : Generator, optional
        Random generator. Same seed -> same output.
    out : Tensor, optional
        Not supported; ``out=`` raises ``NotImplementedError``.
    dtype : torch.dtype, optional
        Floating-point dtype. Default: torch default dtype.
    layout : torch.layout, optional
        Must be ``torch.strided``.
    device : torch.device, optional
        Default is CPU.
    requires_grad : bool, optional
        Default is False.
    pin_memory : bool, optional
        Default is False.
    memory_format : torch.memory_format, optional
        Must be ``torch.contiguous_format``.

    Returns
    -------
    Tensor
        The generated violet noise, shape ``(size,)``.

    Examples
    --------
    >>> violet_noise(10, generator=torch.Generator(device="cpu").manual_seed(1))
    """
    return dispatch_colored_noise(
        "violet_noise",
        torch.ops.torchscience.violet_noise,
        size,
        generator=generator,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
    )
