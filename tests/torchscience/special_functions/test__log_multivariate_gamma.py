import functools

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
)


class TestLogMultivariateGamma(OpTestCase):
    """Tests for the log_multivariate_gamma function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="log_multivariate_gamma",
            func=functools.partial(
                torchscience.special_functions.log_multivariate_gamma, d=3
            ),
            arity=1,
            input_specs=[
                InputSpec(
                    name="a", position=0, default_real_range=(3.0, 10.0)
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_complex_dtypes",
                "test_dtype_preservation",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_low_precision_dtype_preservation",
                "test_low_precision_forward",
            },
        )
