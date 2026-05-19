"""Rectangular-image (xdim != ydim) regression tests for stochastic_optics."""
import numpy as np
import pytest
import scipy.signal

import ehtim.scattering.stochastic_optics as so


RECT_XDIM = 32
RECT_YDIM = 48


@pytest.fixture(scope="module")
def model():
    """Default Sgr A*-like dipole scattering model."""
    return so.ScatteringModel(model="dipole")


class TestSqrtQMatrix:
    """sqrtQ_Matrix shape + finiteness on rectangular reference images."""

    def test_shape_matches_rect_reference_image(self, model, make_rect_image):
        """sqrtQ has shape (ydim, xdim) matching the reference image."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        sqrtQ = model.sqrtQ_Matrix(im)
        assert sqrtQ.shape == (im.ydim, im.xdim)

    def test_returns_finite_values(self, model, make_rect_image):
        """The Fourier-domain power spectrum is finite everywhere on rect."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        sqrtQ = model.sqrtQ_Matrix(im)
        assert np.all(np.isfinite(sqrtQ))


class TestEnsembleAverageKernel:
    """Ensemble_Average_Kernel normalization on rectangular reference images."""

    def test_kernel_sums_to_one(self, model, make_rect_image):
        """The ensemble-average kernel is flux-preserving (sums to 1)."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        ker = model.Ensemble_Average_Kernel(im)
        assert np.sum(ker) == pytest.approx(1.0, rel=1e-10)


class TestWrappedConvolve:
    """Wrapped_Convolve helper agrees with scipy.signal.fftconvolve."""

    def test_matches_scipy_on_square_inputs(self):
        """On square inputs the helper matches scipy fftconvolve to numerical precision."""
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((16, 16))
        ker = rng.standard_normal((16, 16))
        out = so.Wrapped_Convolve(sig, ker)
        ref = scipy.signal.fftconvolve(
            np.pad(sig, ((16, 16), (16, 16)), "wrap"),
            np.pad(ker, ((16, 16), (16, 16)), "constant"),
            mode="same",
        )[16:32, 16:32]
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)
