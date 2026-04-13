"""Tests for ehtim diagnostics module."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import ehtim.diagnostics as ds


def gauss(x0, sx):
    n = 128
    X = 0.5 * (n - 1)
    x = np.linspace(-X, X, n) / n - x0
    return np.exp(-0.5 * (x * x) / (sx * sx)) / np.sqrt(2 * np.pi * sx * sx)


def gauss2(x0, y0, sx, sy):
    return np.sqrt(np.outer(gauss(x0, sx), gauss(y0, sy)))


class TestOnedimize:
    """Test the onedimize() diagnostic function."""

    def test_output_shapes(self):
        imgs = [gauss2(0.0, 0.0, 0.1, 0.1)]
        oneds, ref = ds.onedimize(imgs, n=32)
        assert len(oneds) == 1
        assert ref.shape == (32 * 32,)
        assert oneds[0].shape == (32 * 32,)

    def test_multiple_images(self):
        imgs = [
            gauss2(0.0, 0.0, 0.1, 0.1),
            gauss2(0.0, -0.1, 0.1, 0.1),
            gauss2(0.0, -0.2, 0.1, 0.1),
        ]
        oneds, ref = ds.onedimize(imgs, n=32)
        assert len(oneds) == len(imgs)

    def test_ref_is_sorted_descending(self):
        imgs = [gauss2(0.0, 0.0, 0.1, 0.1)]
        oneds, ref = ds.onedimize(imgs, n=32)
        assert np.all(ref[:-1] >= ref[1:])
