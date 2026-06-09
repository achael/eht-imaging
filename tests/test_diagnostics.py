"""Tests for ehtim diagnostics module."""

import numpy as np

import ehtim.diagnostics as ds

# Sizes for the sumdown tests. The 1D length is non-trivial (not a multiple
# of N) so the piecewise-linear cumulative bin boundaries are exercised; the
# image padding path likewise needs nx, ny not multiples of N.
SUMDOWN_LINE_LEN = 100
SUMDOWN_IMG_SHAPE = (45, 37)
SUMDOWN_N = 16
SUMDOWN_TOTAL_RTOL = 1e-10


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


class TestSumdownLin:
    """Test the sumdown_lin() 1D segment summer."""

    def test_output_shape(self):
        out = ds.sumdown_lin(np.zeros(SUMDOWN_LINE_LEN), n=SUMDOWN_N)
        assert out.shape == (SUMDOWN_N,)

    def test_preserves_total(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(SUMDOWN_LINE_LEN)
        out = ds.sumdown_lin(y, n=SUMDOWN_N)
        np.testing.assert_allclose(out.sum(), y.sum(), rtol=SUMDOWN_TOTAL_RTOL)

    def test_uniform_input_gives_uniform_output(self):
        y = np.ones(SUMDOWN_LINE_LEN)
        out = ds.sumdown_lin(y, n=SUMDOWN_N)
        expected = SUMDOWN_LINE_LEN / SUMDOWN_N
        np.testing.assert_allclose(out, expected, rtol=1e-10)


class TestSumdownImg:
    """Test the sumdown_img() 2D block summer."""

    def test_output_shape(self):
        out = ds.sumdown_img(np.zeros(SUMDOWN_IMG_SHAPE), n=SUMDOWN_N)
        assert out.shape == (SUMDOWN_N, SUMDOWN_N)

    def test_preserves_total(self):
        rng = np.random.default_rng(1)
        img = rng.standard_normal(SUMDOWN_IMG_SHAPE)
        out = ds.sumdown_img(img, n=SUMDOWN_N)
        # Padding fills with zeros, so the total is exactly conserved.
        np.testing.assert_allclose(out.sum(), img.sum(), rtol=SUMDOWN_TOTAL_RTOL)

    def test_uniform_image_gives_uniform_output(self):
        img = np.ones(SUMDOWN_IMG_SHAPE)
        out = ds.sumdown_img(img, n=SUMDOWN_N)
        # Inner cells are full mx*my cells of 1; edge cells contain padding
        # zeros. Total still equals nx*ny.
        np.testing.assert_allclose(out.sum(),
                                   SUMDOWN_IMG_SHAPE[0] * SUMDOWN_IMG_SHAPE[1])
