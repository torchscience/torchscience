from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def sine_wave(
    n: int,
    frequency: float = 1.0,
    sample_rate: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Sine wave generator.

    Generates a discrete sinusoidal waveform of n samples.

    Mathematical Definition
    -----------------------
    The sine wave is defined as:

        y[k] = A * sin(2 * pi * f * k / fs + phi)

    where:
        - A is the amplitude
        - f is the frequency in Hz (or cycles per unit time)
        - fs is the sample rate in samples per unit time
        - phi is the initial phase in radians
        - k is the sample index (0, 1, 2, ..., n-1)

    The normalized frequency (f/fs) determines how many cycles occur per sample.
    For a frequency of 1.0 Hz and sample rate of 1.0, one complete cycle spans
    one sample (i.e., the Nyquist limit).

    Parameters
    ----------
    n : int
        Number of samples in the output waveform. Must be non-negative.
        If n=0, an empty tensor is returned.
    frequency : float, optional
        Frequency of the sine wave in Hz (cycles per unit time). Default is 1.0.
        The normalized frequency is computed as frequency/sample_rate.
    sample_rate : float, optional
        Sampling rate in samples per unit time. Default is 1.0.
        Must be positive.
    amplitude : float, optional
        Peak amplitude of the sine wave. Default is 1.0.
        The output will range from -amplitude to +amplitude.
    phase : float, optional
        Initial phase offset in radians. Default is 0.0.
        A phase of pi/2 produces a cosine wave.
    dtype : torch.dtype, optional
        The desired data type of the returned tensor. If None, uses the
        default floating point type.
    layout : torch.layout, optional
        The desired layout of the returned tensor. If None, uses the
        default layout (torch.strided).
    device : torch.device, optional
        The desired device of the returned tensor. If None, uses the
        default device.
    requires_grad : bool, optional
        If True, the returned tensor will require gradients. Default is False.

    Returns
    -------
    Tensor
        A 1-D tensor of size (n,) containing the sine wave samples.

    Examples
    --------
    Generate a basic sine wave with 100 samples:

    >>> sine_wave(100)
    tensor([0.0000, 0.0628, 0.1253, ...])

    Generate a 440 Hz tone sampled at 44100 Hz for 1 second:

    >>> samples = sine_wave(44100, frequency=440.0, sample_rate=44100.0)

    Generate a cosine wave using phase offset:

    >>> import math
    >>> cosine = sine_wave(100, phase=math.pi / 2)

    Generate with specific dtype:

    >>> sine_wave(10, dtype=torch.float64)
    tensor([0.0000, 0.0628, ...], dtype=torch.float64)

    Raises
    ------
    RuntimeError
        If n < 0 or sample_rate <= 0.

    See Also
    --------
    torch.sin : Element-wise sine function
    torch.arange : Create a range tensor

    Notes
    -----
    Aliasing Considerations
    ^^^^^^^^^^^^^^^^^^^^^^^
    To avoid aliasing artifacts, the frequency should be less than half the
    sample rate (Nyquist frequency). When frequency >= sample_rate/2, the
    waveform will exhibit aliasing.

    Frequency Resolution
    ^^^^^^^^^^^^^^^^^^^^
    The frequency resolution depends on the number of samples and sample rate.
    For precise frequency representation in spectral analysis, ensure that the
    number of samples is an integer multiple of the period (sample_rate/frequency).

    References
    ----------
    A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal Processing,"
    3rd ed., Prentice Hall, 2009.

    J.O. Smith III, "Mathematics of the Discrete Fourier Transform (DFT),"
    W3K Publishing, 2007.
    """
    return torch.ops.torchscience.sine_wave(
        n,
        frequency,
        sample_rate,
        amplitude,
        phase,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
