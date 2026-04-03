import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestAiryAi(OpTestCase):
    """Tests for the airy_ai function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="airy_ai",
            func=torchscience.special_functions.airy_ai,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z", position=0, default_real_range=(-5.0, 5.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=0.35502805388781724,
                    description="Ai(0) = 1/(3^(2/3) * Gamma(2/3))",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=0.13529241631288141,
                    description="Ai(1)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=0.03492413042327638,
                    description="Ai(2)",
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
            },
        )
