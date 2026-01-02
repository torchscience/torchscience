"""Tests for FiniteDifferenceStencil."""


class TestStencilImport:
    """Tests for stencil imports."""

    def test_stencil_importable(self):
        """FiniteDifferenceStencil is importable."""
        from torchscience.differentiation import FiniteDifferenceStencil

        assert FiniteDifferenceStencil is not None

    def test_exceptions_importable(self):
        """Exceptions are importable."""
        from torchscience.differentiation import StencilError

        assert StencilError is not None
