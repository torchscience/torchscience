import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import sin_pi, cos_pi


class TestCosPi(UnaryOperatorTestCase):
    func = staticmethod(cos_pi)
    op_name = "_cos_pi"

    symmetry = "even"
    period = 2.0
    bounds = (-1.0, 1.0)

    known_values = {
        0.0: 1.0,
        0.5: 0.0,
        1.0: -1.0,
        1.5: 0.0,
        2.0: 1.0,
    }
    zeros = [-1.5, -0.5, 0.5, 1.5, 2.5]

    reference = staticmethod(lambda x: torch.cos(torch.pi * x))

    # Our implementation is more accurate at special values (half-integers)
    reference_atol = 1e-5
    reference_rtol = 1e-5

    identities = [
        (lambda x: sin_pi(x) ** 2 + cos_pi(x) ** 2, 1.0),  # Pythagorean identity
    ]
