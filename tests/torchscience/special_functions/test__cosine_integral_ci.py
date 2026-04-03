import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestCosineIntegralCi(OpTestCase):
    """Tests for the cosine integral Ci function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="cosine_integral_ci",
            func=torchscience.special_functions.cosine_integral_ci,
            arity=1,
            input_specs=[
                InputSpec(
                    name="x",
                    position=0,
                    default_real_range=(0.5, 10.0),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.5,),
                    expected=-0.17778407877534814,
                    description="Ci(0.5)",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=0.33740392290096813,
                    description="Ci(1.0)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=0.42298082631138887,
                    description="Ci(2.0)",
                ),
                SpecialValue(
                    inputs=(5.0,),
                    expected=-0.19002974965664387,
                    description="Ci(5.0)",
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
                "test_low_precision_forward",
                "test_special_values",
            },
        )
