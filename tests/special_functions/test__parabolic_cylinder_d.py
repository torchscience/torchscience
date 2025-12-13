import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import parabolic_cylinder_d


class TestParabolicCylinderD(BinaryOperatorTestCase):
    func = staticmethod(parabolic_cylinder_d)
    op_name = "_parabolic_cylinder_d"

    # Known values from mathematical tables
    # D_0(0) = 1
    # D_1(0) = 0
    # D_{-1}(0) = sqrt(pi/2)
    known_values = [
        ((0.0, 0.0), 1.0),
        ((1.0, 0.0), 0.0),
    ]

    # Reference implementation using scipy
    reference = staticmethod(lambda nu, z: torch.from_numpy(
        scipy.special.pbdv(nu.numpy(), z.numpy())[0]
    ).to(nu.dtype))

    # Input ranges for Hypothesis
    input_range_1 = (-5.0, 5.0)  # nu
    input_range_2 = (-5.0, 5.0)  # z

    # Gradcheck inputs (use values away from singularities)
    gradcheck_inputs = ([0.5, 1.0, 1.5], [0.5, 1.0, 2.0])

    supports_complex = False
