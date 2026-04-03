import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestFresnelS(OpTestCase):
    """Tests for the Fresnel sine integral S(z)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="fresnel_s",
            func=torchscience.special_functions.fresnel_s,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z",
                    position=0,
                    default_real_range=(-5.0, 5.0),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=0.0,
                    description="S(0) = 0",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=0.4382591473903548,
                    description="S(1) ~ 0.4382591473903548",
                ),
                SpecialValue(
                    inputs=(float("inf"),),
                    expected=0.5,
                    description="S(+inf) = 0.5",
                ),
                SpecialValue(
                    inputs=(float("-inf"),),
                    expected=-0.5,
                    description="S(-inf) = -0.5",
                ),
            ],
            skip_tests={
                "test_gradcheck_complex",
                "test_gradcheck_real",
                "test_gradgradcheck_complex",
                "test_low_precision_forward",
            },
        )
