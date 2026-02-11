import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestGammaSign(OpTestCase):
    """Tests for the gamma_sign function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="gamma_sign",
            func=torchscience.special_functions.gamma_sign,
            arity=1,
            input_specs=[
                InputSpec(name="x", position=0, default_real_range=(0.5, 5.0)),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0,),
                    expected=1.0,
                    description="gamma_sign(1) = 1",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=1.0,
                    description="gamma_sign(2) = 1",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=1.0,
                    description="gamma_sign(0.5) = 1",
                ),
            ],
            skip_tests={
                "test_gradcheck_real",
                "test_gradgradcheck_real",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
            },
        )
