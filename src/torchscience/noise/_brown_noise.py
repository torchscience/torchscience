import torch
from torch import Tensor

from ._common import dispatch_colored_noise


def brown_noise(
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
    Brown(ian) noise.

    Generates brown noise (also called red noise or Brownian noise) of the
    given size. Brown noise has a power spectral density that falls as
    :math:`1/f^2`:

    .. math::
        S(f) \propto \frac{1}{f^2}.

    Power per octave halves for each octave shift. This is the spectral
    analogue of a band-limited Wiener (Brownian) process: the running
    integral of white noise has the same 1/f^2 shape, hence the name.

    Algorithm
    ---------

    1. Draw N i.i.d. samples from N(0, 1).
    2. Take the real FFT.
    3. Divide each non-DC bin by ``|f|``, giving |X(f)| ~ 1/f and hence
       |X(f)|^2 ~ 1/f^2.
    4. Set the DC bin (f=0) to zero -- 1/f^2 is undefined at f=0, and
       zeroing DC makes the time-domain output exactly zero-mean.
    5. Inverse FFT, then normalize by max absolute value so peak amplitude
       is 1.

    Testable Guarantees
    -------------------
    - PSD log-log slope equals -2 (slope test).
    - Power halves per octave (octave-power-ratio test).
    - Mean is zero up to FFT round-off.
    - Peak absolute value is exactly 1.

    Applications
    ------------
    - Modeling Brownian / random-walk-like processes in a stationary,
      band-limited setting.
    - Geophysical and climate-noise simulation, where 1/f^2 spectra are
      common.

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
        The generated brown noise, shape ``(size,)``.

    Examples
    --------
    >>> brown_noise(10, generator=torch.Generator(device="cpu").manual_seed(1))
    """
    return dispatch_colored_noise(
        "brown_noise",
        torch.ops.torchscience.brown_noise,
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
