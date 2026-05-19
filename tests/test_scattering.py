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
    """Ensemble_Average_Kernel shape + normalization on rectangular reference images."""

    def test_shape_matches_rect_reference_image(self, model, make_rect_image):
        """The blurring kernel has shape (ydim, xdim) matching the reference image."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        ker = model.Ensemble_Average_Kernel(im)
        assert ker.shape == (im.ydim, im.xdim)

    def test_kernel_sums_to_one(self, model, make_rect_image):
        """The ensemble-average kernel is flux-preserving (sums to 1)."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        ker = model.Ensemble_Average_Kernel(im)
        assert np.sum(ker) == pytest.approx(1.0, rel=1e-10)


class TestMakePhaseScreen:
    """MakePhaseScreen shape contract on rectangular reference images."""

    def test_shape_matches_rect_epsilon_screen(self, model, make_rect_image):
        """Phase screen output has the (ydim, xdim) shape of the reference image."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        eps = so.MakeEpsilonScreen(im.xdim, im.ydim, rngseed=1)
        phi = model.MakePhaseScreen(eps, im)
        assert phi.imvec.shape == (im.ydim * im.xdim,)
        assert np.all(np.isfinite(phi.imvec))


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

    def test_preserves_shape_on_rect_inputs(self):
        """Wrapped_Convolve on rect (ydim, xdim) inputs returns matching-shape output."""
        rng = np.random.default_rng(1)
        sig = rng.standard_normal((RECT_YDIM, RECT_XDIM))
        ker = rng.standard_normal((RECT_YDIM, RECT_XDIM))
        out = so.Wrapped_Convolve(sig, ker)
        assert out.shape == sig.shape

    def test_matches_scipy_on_rect_inputs(self):
        """On rect inputs the helper matches the equivalent scipy fftconvolve form."""
        rng = np.random.default_rng(2)
        sig = rng.standard_normal((RECT_YDIM, RECT_XDIM))
        ker = rng.standard_normal((RECT_YDIM, RECT_XDIM))
        out = so.Wrapped_Convolve(sig, ker)
        ref = scipy.signal.fftconvolve(
            np.pad(sig, ((RECT_YDIM, RECT_YDIM), (RECT_XDIM, RECT_XDIM)), "wrap"),
            np.pad(ker, ((RECT_YDIM, RECT_YDIM), (RECT_XDIM, RECT_XDIM)), "constant"),
            mode="same",
        )[RECT_YDIM:(2*RECT_YDIM), RECT_XDIM:(2*RECT_XDIM)]
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)


class TestMakeEpsilonScreenFromList:
    """MakeEpsilonScreenFromList runs on modern NumPy (no np.complex deprecation)."""

    def test_returns_complex_square_screen(self):
        """The function returns a complex (N, N) array; square-only by design."""
        n = 9
        eps_list = [0.0] * (n * n - 1)
        eps = so.MakeEpsilonScreenFromList(eps_list, n)
        assert eps.shape == (n, n)
        assert np.iscomplexobj(eps)
