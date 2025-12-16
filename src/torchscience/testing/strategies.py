import math
from typing import Optional, Set, Tuple

import hypothesis.extra.numpy
import hypothesis.strategies
import numpy
import torch

# =============================================================================
# Numeric Value Strategies
# =============================================================================


def positive_reals(
    min_value: float = 1e-10,
    max_value: float = 1e10,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for positive real numbers."""
    return hypothesis.strategies.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


def negative_reals(
    min_value: float = -1e10,
    max_value: float = -1e-10,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for negative real numbers."""
    return hypothesis.strategies.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


def reals(
    min_value: float = -1e10,
    max_value: float = 1e10,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for real numbers."""
    return hypothesis.strategies.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


def non_integer_reals(
    min_value: float = -100.0,
    max_value: float = 100.0,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for real numbers that are not integers."""
    return hypothesis.strategies.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    ).filter(lambda x: x != math.floor(x))


def integers_as_floats(
    min_value: int = -100,
    max_value: int = 100,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for integers represented as floats."""
    return hypothesis.strategies.integers(
        min_value=min_value, max_value=max_value
    ).map(float)


def chebyshev_domain(
    include_endpoints: bool = False,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for values in [-1, 1] (Chebyshev domain)."""
    if include_endpoints:
        return hypothesis.strategies.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    return hypothesis.strategies.floats(
        min_value=-0.999,
        max_value=0.999,
        allow_nan=False,
        allow_infinity=False,
    )


def hyperbolic_domain(
    min_value: float = 1.001,
    max_value: float = 100.0,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for values > 1 (hyperbolic continuation domain)."""
    return hypothesis.strategies.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


def avoiding_values(
    excluded: Set[float],
    min_value: float = -100.0,
    max_value: float = 100.0,
    min_distance: float = 0.01,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for values avoiding specified points."""
    return hypothesis.strategies.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    ).filter(lambda x: all(abs(x - e) > min_distance for e in excluded))


def avoiding_poles(
    max_negative_pole: int = -100,
    min_value: float = -100.0,
    max_value: float = 100.0,
    min_distance: float = 0.01,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for values avoiding poles at non-positive integers."""
    poles = set(range(max_negative_pole, 1))
    return avoiding_values(poles, min_value, max_value, min_distance)


# =============================================================================
# Complex Number Strategies
# =============================================================================


@hypothesis.strategies.composite
def complex_numbers(
    draw: hypothesis.strategies.DrawFn,
    real_range: Tuple[float, float] = (-10.0, 10.0),
    imag_range: Tuple[float, float] = (-10.0, 10.0),
) -> complex:
    """Strategy for complex numbers."""
    real = draw(
        hypothesis.strategies.floats(
            min_value=real_range[0],
            max_value=real_range[1],
            allow_nan=False,
            allow_infinity=False,
        )
    )
    imag = draw(
        hypothesis.strategies.floats(
            min_value=imag_range[0],
            max_value=imag_range[1],
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return complex(real, imag)


@hypothesis.strategies.composite
def complex_avoiding_real_axis(
    draw: hypothesis.strategies.DrawFn,
    real_range: Tuple[float, float] = (-10.0, 10.0),
    min_imag: float = 0.1,
    max_imag: float = 10.0,
) -> complex:
    """Strategy for complex numbers with non-zero imaginary part."""
    real = draw(
        hypothesis.strategies.floats(
            min_value=real_range[0],
            max_value=real_range[1],
            allow_nan=False,
            allow_infinity=False,
        )
    )
    imag_magnitude = draw(
        hypothesis.strategies.floats(
            min_value=min_imag,
            max_value=max_imag,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    sign = draw(hypothesis.strategies.sampled_from([-1, 1]))
    return complex(real, sign * imag_magnitude)


# =============================================================================
# Tensor Strategies
# =============================================================================


@hypothesis.strategies.composite
def shapes(
    draw: hypothesis.strategies.DrawFn,
    min_dims: int = 0,
    max_dims: int = 4,
    min_side: int = 1,
    max_side: int = 10,
) -> Tuple[int, ...]:
    """Strategy for tensor shapes."""
    ndims = draw(
        hypothesis.strategies.integers(min_value=min_dims, max_value=max_dims)
    )
    if ndims == 0:
        return ()
    return tuple(
        draw(
            hypothesis.strategies.integers(
                min_value=min_side, max_value=max_side
            )
        )
        for _ in range(ndims)
    )


@hypothesis.strategies.composite
def tensors(
    draw: hypothesis.strategies.DrawFn,
    dtype: torch.dtype = torch.float64,
    shape: Optional[Tuple[int, ...]] = None,
    min_dims: int = 1,
    max_dims: int = 4,
    min_side: int = 1,
    max_side: int = 10,
    elements: Optional[hypothesis.strategies.SearchStrategy[float]] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate random tensors with configurable properties."""
    if shape is None:
        shape = draw(shapes(min_dims, max_dims, min_side, max_side))

    if elements is None:
        elements = hypothesis.strategies.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        )

    if dtype in (torch.complex64, torch.complex128):
        # Generate complex tensors
        real_np_dtype = (
            numpy.float32 if dtype == torch.complex64 else numpy.float64
        )
        real_arr = draw(
            hypothesis.extra.numpy.arrays(
                real_np_dtype, shape, elements=elements
            )
        )
        imag_arr = draw(
            hypothesis.extra.numpy.arrays(
                real_np_dtype, shape, elements=elements
            )
        )
        tensor = torch.complex(
            torch.tensor(
                real_arr,
                dtype=torch.float32
                if dtype == torch.complex64
                else torch.float64,
            ),
            torch.tensor(
                imag_arr,
                dtype=torch.float32
                if dtype == torch.complex64
                else torch.float64,
            ),
        )
    else:
        # Generate real tensors
        np_dtype = {
            torch.float16: numpy.float16,
            torch.bfloat16: numpy.float32,  # Use float32 for generation, cast later
            torch.float32: numpy.float32,
            torch.float64: numpy.float64,
        }.get(dtype, numpy.float64)
        arr = draw(
            hypothesis.extra.numpy.arrays(np_dtype, shape, elements=elements)
        )
        tensor = torch.tensor(arr, dtype=dtype)

    return tensor.to(device)


@hypothesis.strategies.composite
def broadcastable_shapes(
    draw: hypothesis.strategies.DrawFn,
    max_dims: int = 4,
    max_side: int = 10,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Generate pairs of shapes that can be broadcast together."""
    # Generate the result shape first
    result_ndims = draw(
        hypothesis.strategies.integers(min_value=1, max_value=max_dims)
    )
    result_shape = tuple(
        draw(hypothesis.strategies.integers(min_value=1, max_value=max_side))
        for _ in range(result_ndims)
    )

    # Generate first shape by potentially dropping/keeping dimensions
    shape1 = []
    for dim in result_shape:
        choice = draw(
            hypothesis.strategies.sampled_from(["keep", "one", "drop"])
        )
        if choice == "keep":
            shape1.append(dim)
        elif choice == "one":
            shape1.append(1)
        # "drop" means we don't add this dimension

    # Generate second shape similarly
    shape2 = []
    for dim in result_shape:
        choice = draw(
            hypothesis.strategies.sampled_from(["keep", "one", "drop"])
        )
        if choice == "keep":
            shape2.append(dim)
        elif choice == "one":
            shape2.append(1)

    # Ensure at least one element in each shape
    if not shape1:
        shape1 = [1]
    if not shape2:
        shape2 = [1]

    return tuple(shape1), tuple(shape2)


@hypothesis.strategies.composite
def broadcastable_tensor_pairs(
    draw: hypothesis.strategies.DrawFn,
    dtype: torch.dtype = torch.float64,
    max_dims: int = 4,
    max_side: int = 10,
    elements: Optional[hypothesis.strategies.SearchStrategy[float]] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate pairs of tensors that can be broadcast together."""
    shape1, shape2 = draw(broadcastable_shapes(max_dims, max_side))
    t1 = draw(
        tensors(dtype=dtype, shape=shape1, elements=elements, device=device)
    )
    t2 = draw(
        tensors(dtype=dtype, shape=shape2, elements=elements, device=device)
    )
    return t1, t2


# =============================================================================
# Mathematical Property Strategies
# =============================================================================


@hypothesis.strategies.composite
def recurrence_test_values(
    draw: hypothesis.strategies.DrawFn,
    base_range: Tuple[float, float] = (-0.99, 0.99),
    max_n: int = 20,
) -> Tuple[torch.Tensor, int]:
    """Generate values for testing recurrence relations."""
    z = draw(
        hypothesis.strategies.floats(
            min_value=base_range[0],
            max_value=base_range[1],
            allow_nan=False,
            allow_infinity=False,
        )
    )
    n = draw(hypothesis.strategies.integers(min_value=2, max_value=max_n))
    return torch.tensor([z], dtype=torch.float64), n


@hypothesis.strategies.composite
def reflection_test_values(
    draw: hypothesis.strategies.DrawFn,
    avoid_integers: bool = True,
) -> float:
    """Generate values for testing reflection formulas."""
    x = draw(
        hypothesis.strategies.floats(
            min_value=0.01,
            max_value=0.99,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    if avoid_integers:
        # Ensure we're not too close to integers
        return x
    return x


# =============================================================================
# Dtype Strategies
# =============================================================================


real_dtypes = hypothesis.strategies.sampled_from(
    [torch.float32, torch.float64]
)
complex_dtypes = hypothesis.strategies.sampled_from(
    [torch.complex64, torch.complex128]
)
low_precision_dtypes = hypothesis.strategies.sampled_from(
    [torch.float16, torch.bfloat16]
)
all_floating_dtypes = hypothesis.strategies.sampled_from(
    [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
all_complex_dtypes = hypothesis.strategies.sampled_from(
    [torch.complex64, torch.complex128]
)
all_dtypes = hypothesis.strategies.sampled_from(
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ]
)


# =============================================================================
# Device Strategies
# =============================================================================


def available_devices() -> hypothesis.strategies.SearchStrategy[str]:
    """Strategy for available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return hypothesis.strategies.sampled_from(devices)
