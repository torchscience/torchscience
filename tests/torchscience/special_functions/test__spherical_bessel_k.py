import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestSphericalBesselK(OpTestCase):
    """Tests for the modified spherical Bessel function k_n(z)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="spherical_bessel_k",
            func=torchscience.special_functions.spherical_bessel_k,
            arity=2,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="z", position=1, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.0),
                    expected=float("inf"),
                    description="k_0(0) = +inf",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0),
                    expected=float("inf"),
                    description="k_1(0) = +inf",
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
            },
        )
