import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
)


class TestSphericalHankel2(OpTestCase):
    """Tests for the spherical Hankel function of the second kind h_n^(2)(z)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="spherical_hankel_2",
            func=torchscience.special_functions.spherical_hankel_2,
            arity=2,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="z", position=1, default_real_range=(0.5, 10.0)
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_dtype_preservation",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_dtype_preservation",
                "test_low_precision_forward",
                "test_real_dtypes",
            },
        )
