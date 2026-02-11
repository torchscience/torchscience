import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestKrawtchoukPolynomialK(OpTestCase):
    """Tests for the krawtchouk_polynomial_k function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="krawtchouk_polynomial_k",
            func=torchscience.special_functions.krawtchouk_polynomial_k,
            arity=4,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 3.0)),
                InputSpec(name="x", position=1, default_real_range=(0.1, 3.0)),
                InputSpec(name="p", position=2, default_real_range=(0.1, 0.9)),
                InputSpec(
                    name="N", position=3, default_real_range=(3.0, 10.0)
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_compile_smoke",
                "test_cpu_device",
                "test_gradcheck_complex",
                "test_gradcheck_real",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
                "test_vmap_over_batch",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 1.0, 0.5, 5.0),
                    expected=1.0,
                    description="K_0(x; p, N) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0, 0.5, 5.0),
                    expected=1.0,
                    description="K_1(0; p, N) = 1 - 0/(N*p) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 1.0, 0.5, 5.0),
                    expected=0.6,
                    description="K_1(1; 0.5, 5) = 1 - 1/(5*0.5) = 0.6",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
