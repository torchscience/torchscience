import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestModifiedBesselK(OpTestCase):
    """Tests for the modified Bessel function of the second kind K_n(z)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="modified_bessel_k",
            func=torchscience.special_functions.modified_bessel_k,
            arity=2,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="z", position=1, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, float("inf")),
                    expected=0.0,
                    description="K_0(+inf) = 0",
                ),
                SpecialValue(
                    inputs=(1.0, float("inf")),
                    expected=0.0,
                    description="K_1(+inf) = 0",
                ),
                SpecialValue(
                    inputs=(0.0, 0.0),
                    expected=float("inf"),
                    description="K_0(0) = +inf",
                ),
            ],
        )
