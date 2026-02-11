import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestAiryBi(OpTestCase):
    """Tests for the airy_bi function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="airy_bi",
            func=torchscience.special_functions.airy_bi,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z", position=0, default_real_range=(-5.0, 5.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=0.61492662744600073515,
                    description="Bi(0) = 1/(3^(1/6) * Gamma(2/3))",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=1.2074235949528713,
                    description="Bi(1)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=3.2980949999782147,
                    description="Bi(2)",
                ),
            ],
        )
