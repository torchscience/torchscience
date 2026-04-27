import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - registers torch.ops.torchscience


def grey_noise(
    size: int,
    *,
    sample_rate: float = 44100.0,
    generator=None,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad: bool = False,
    pin_memory: bool = False,
    memory_format=torch.contiguous_format,
) -> Tensor:
    r"""
    Grey noise.

    Generates grey noise of the given size: white noise pre-emphasized by the
    inverse of the IEC 61672-1 A-weighting curve so that, when re-A-weighted
    (e.g. by the human auditory system), the perceived spectrum is
    approximately flat. Equivalently, the power spectral density is

    .. math::
        S(f) = \frac{C}{|H_A(f)|^2},

    where :math:`H_A(f)` is the A-weighting amplitude response. Grey noise is
    the audio-engineering "perceptually flat" counterpart to white noise.

    A-weighting Curve
    -----------------

    The A-weighting magnitude response (un-offset) is

    .. math::
        R_A(f) = \frac{12194^2 \cdot f^4}
                     {(f^2 + 20.6^2) \, \sqrt{(f^2 + 107.7^2)(f^2 + 737.9^2)}
                      \, (f^2 + 12194^2)}.

    The standard +2.00 dB normalization that makes :math:`A(1000) = 0\ \mathrm{dB}`
    is a constant overall amplitude scaling that drops out under this
    function's max-abs normalization, so it is omitted in the kernel.

    Algorithm
    ---------

    1. Draw N i.i.d. samples from N(0, 1).
    2. Take the real FFT.
    3. Multiply each non-DC bin by ``1 / R_A(f)``.
    4. Set the DC bin to zero -- ``R_A(0) = 0`` (the formula has an
       :math:`f^4` numerator factor) so ``1/R_A`` is undefined at DC, and
       zeroing also makes the output exactly zero-mean.
    5. Inverse FFT, then normalize by max absolute value so peak amplitude
       is 1.

    Testable Guarantees
    -------------------
    - The A-weighted PSD is approximately flat in the audible band.
    - Power at low frequencies (where A-weighting attenuates strongly) is
      boosted by approximately ``1 / R_A(f)^2`` relative to the 1 kHz
      reference.
    - Mean is zero up to FFT round-off.
    - Peak absolute value is exactly 1.

    Applications
    ------------
    - Audio system testing where a perceptually-flat reference noise is
      desired (microphone calibration, listening tests).
    - Hearing-test stimuli that compensate for the equal-loudness contour
      of the auditory system.

    Dtype Support
    -------------
    - Supports float16, bfloat16, float32, float64.

    Parameters
    ----------
    size : int
        Length of the output 1-D tensor.
    sample_rate : float, optional
        Sample rate in Hz. Determines the absolute frequency grid that the
        A-weighting curve is evaluated on. Default is 44100.0 (CD-quality
        audio).
    generator : Generator, optional
        Random generator. Same seed -> same output (for the same
        ``sample_rate``).
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
        The generated grey noise, shape ``(size,)``.

    Examples
    --------
    >>> grey_noise(10, generator=torch.Generator(device="cpu").manual_seed(1))
    """
    if out is not None:
        raise NotImplementedError("grey_noise: out= is not supported")
    if layout != torch.strided:
        raise ValueError("grey_noise: only strided layout is supported")
    if memory_format != torch.contiguous_format:
        raise ValueError("grey_noise: only contiguous_format is supported")
    if not (sample_rate > 0):
        raise ValueError(
            f"grey_noise: sample_rate must be positive, got {sample_rate}"
        )

    dev = torch.device(device) if device is not None else torch.device("cpu")
    dt = dtype if dtype is not None else torch.get_default_dtype()
    anchor_kw = {}
    if pin_memory and dev.type == "cpu":
        anchor_kw["pin_memory"] = True
    anchor = torch.empty((), device=dev, dtype=dt, **anchor_kw)

    return torch.ops.torchscience.grey_noise(
        anchor,
        size,
        float(sample_rate),
        generator,
        dtype,
        layout,
        device,
        requires_grad,
        pin_memory,
    )
