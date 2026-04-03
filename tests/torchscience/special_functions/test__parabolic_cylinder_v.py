import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
)


class TestParabolicCylinderV(OpTestCase):
    """Tests for the parabolic_cylinder_v function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="parabolic_cylinder_v",
            func=torchscience.special_functions.parabolic_cylinder_v,
            arity=2,
            input_specs=[
                InputSpec(
                    name="a",
                    position=0,
                    default_real_range=(-3.0, 3.0),
                ),
                InputSpec(
                    name="x",
                    position=1,
                    default_real_range=(0.5, 5.0),
                ),
            ],
            skip_tests={
                "test_gradcheck_complex",
                "test_gradcheck_real",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
            },
        )
