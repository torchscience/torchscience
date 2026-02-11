import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestFresnelC(OpTestCase):
    """Tests for the Fresnel cosine integral C(z)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="fresnel_c",
            func=torchscience.special_functions.fresnel_c,
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
                    description="C(0) = 0",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=0.7798934003768228,
                    description="C(1) ~ 0.7798934003768228",
                ),
                SpecialValue(
                    inputs=(float("inf"),),
                    expected=0.5,
                    description="C(+inf) = 0.5",
                ),
                SpecialValue(
                    inputs=(float("-inf"),),
                    expected=-0.5,
                    description="C(-inf) = -0.5",
                ),
            ],
        )
