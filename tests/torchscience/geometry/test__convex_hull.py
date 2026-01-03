# tests/torchscience/geometry/test__convex_hull.py


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
