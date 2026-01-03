# tests/torchscience/geometry/test__convex_hull.py
import torch

import torchscience  # noqa: F401 - needed to register C++ operators


class TestConvexHullImport:
    """Tests for convex_hull import."""

    def test_convex_hull_importable(self):
        """convex_hull function is importable."""
        from torchscience.geometry import convex_hull

        assert callable(convex_hull)

    def test_convex_hull_class_importable(self):
        """ConvexHull class is importable."""
        from torchscience.geometry import ConvexHull

        assert ConvexHull is not None


class TestConvexHullSchema:
    """Tests for convex_hull operator schema."""

    def test_operator_exists(self):
        """convex_hull operator is registered."""
        assert hasattr(torch.ops.torchscience, "convex_hull")
