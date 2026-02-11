import torch

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestLegendrePolynomialQ(OpTestCase):
    """Tests for the legendre_polynomial_q function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="legendre_polynomial_q",
            func=torchscience.special_functions.legendre_polynomial_q,
            arity=2,
            input_specs=[
                InputSpec(
                    name="x", position=0, default_real_range=(-0.9, 0.9)
                ),
                InputSpec(name="n", position=1, default_real_range=(0.5, 5.0)),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_complex_dtypes",
                "test_compile_smoke",
                "test_dtype_preservation",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.0),
                    expected=0.0,
                    description="Q_0(0) = arctanh(0) = 0",
                ),
                SpecialValue(
                    inputs=(0.0, 1.0),
                    expected=-1.0,
                    description="Q_1(0) = 0*arctanh(0) - 1 = -1",
                ),
                SpecialValue(
                    inputs=(0.5, 0.0),
                    expected=float(torch.arctanh(torch.tensor(0.5)).item()),
                    rtol=1e-7,
                    atol=1e-7,
                    description="Q_0(0.5) = arctanh(0.5)",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
