"""Unit tests for NFFTInfo / FINUFFTPlan correctness.

These exercise the sign + scale conventions of the NFFT wrapper directly,
independent of the higher-level chisq / imaging machinery. The complementary
direct-vs-nfft parity coverage for sample_vis lives in test_obs_simulate.py.
"""
import numpy as np
import pytest

from ehtim.observing.obs_helpers import FINUFFTPlan, NFFTInfo


def _unit_pulse(u, v, _, dom="F"):
    return 1.0 + 0j


def _rng(seed=0):
    return np.random.default_rng(seed)


def _random_uv(n, psize, rng):
    """uv inside the finufft acceptance band ((-0.5, 0.5] in pixel units)."""
    return rng.uniform(-0.4 / psize, 0.4 / psize, size=(n, 2))


@pytest.fixture(scope="module")
def psize():
    return 1e-9


@pytest.fixture(scope="module")
def asymmetric_uv(psize):
    return _random_uv(200, psize, _rng(0))


class TestFINUFFTPlan:
    """Direct tests of FINUFFTPlan's stateful API."""

    def test_forward_adjoint_transpose(self, psize, asymmetric_uv):
        """<A f_hat, g> == <f_hat, A^T g> for random arrays."""
        xdim, ydim = 64, 48
        uv_finufft = 2 * np.pi * asymmetric_uv * psize
        plan = FINUFFTPlan(xdim, ydim, uv_finufft, eps=1e-12)

        rng = _rng(1)
        f_hat = (rng.standard_normal((xdim, ydim))
                 + 1j * rng.standard_normal((xdim, ydim))).astype("complex128")
        g = (rng.standard_normal(len(asymmetric_uv))
             + 1j * rng.standard_normal(len(asymmetric_uv))).astype("complex128")

        plan.f_hat = f_hat
        plan.trafo()
        Af = plan.f.copy()

        plan.f = g
        plan.adjoint()
        ATg = plan.f_hat.copy()

        lhs = np.vdot(Af, g)         # <A f_hat, g>
        rhs = np.vdot(f_hat, ATg)    # <f_hat, A^T g>
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)

    def test_complex_and_real_input(self, psize, asymmetric_uv):
        """trafo accepts float64 f_hat (real images) by upcasting internally."""
        xdim, ydim = 32, 32
        uv_finufft = 2 * np.pi * asymmetric_uv * psize
        plan = FINUFFTPlan(xdim, ydim, uv_finufft)

        real_img = _rng(2).standard_normal((xdim, ydim))
        plan.f_hat = real_img
        plan.trafo()
        out_from_real = plan.f.copy()

        plan.f_hat = real_img.astype("complex128")
        plan.trafo()
        out_from_complex = plan.f.copy()

        np.testing.assert_allclose(out_from_real, out_from_complex, rtol=1e-12, atol=1e-12)


class TestNFFTInfo:
    """End-to-end tests of NFFTInfo against analytic Fourier transforms."""

    def _expected_delta_ft(self, uv, xpix, ypix, xdim, ydim, psize):
        """FT of unit delta at pixel (xpix, ypix), including pulsefac half-shift."""
        u, v = uv[:, 0], uv[:, 1]
        x0 = psize * (xpix - xdim / 2 + 0.5)
        y0 = psize * (ypix - ydim / 2 + 0.5)
        return np.exp(-2j * np.pi * (u * x0 + v * y0))

    def _delta_image(self, xdim, ydim, xpix, ypix):
        img = np.zeros((ydim, xdim))
        img[ypix, xpix] = 1.0
        return img.flatten()

    @pytest.mark.parametrize("xdim,ydim", [(64, 64), (64, 48)])
    def test_centered_delta_matches_analytic_ft(self, xdim, ydim, psize, asymmetric_uv):
        """A delta at an off-centre pixel transforms to the expected phase ramp.

        Catches axis swaps, sign errors, and centering bugs in one shot.
        The rectangular shape (xdim != ydim) catches axis swaps the square case
        cannot see.
        """
        xpix, ypix = 10, 20
        imvec = self._delta_image(xdim, ydim, xpix, ypix)
        info = NFFTInfo(xdim, ydim, psize, _unit_pulse,
                        npad=2 * max(xdim, ydim), p_rad=8, uv=asymmetric_uv,
                        eps=1e-12)
        info.plan.f_hat = imvec.reshape((ydim, xdim)).T
        info.plan.trafo()
        samples = info.plan.f.copy() * info.pulsefac
        expected = self._expected_delta_ft(asymmetric_uv, xpix, ypix,
                                           xdim, ydim, psize)
        np.testing.assert_allclose(samples, expected, rtol=1e-9, atol=1e-10)

    @pytest.mark.parametrize("eps,tol", [(1e-6, 1e-5), (1e-9, 1e-8), (1e-12, 1e-10)])
    def test_eps_accuracy(self, eps, tol, psize, asymmetric_uv):
        """eps controls the relative-accuracy floor of the NFFT output."""
        xdim, ydim = 64, 64
        xpix, ypix = 10, 20
        imvec = self._delta_image(xdim, ydim, xpix, ypix)
        info = NFFTInfo(xdim, ydim, psize, _unit_pulse,
                        npad=128, p_rad=8, uv=asymmetric_uv, eps=eps)
        info.plan.f_hat = imvec.reshape((ydim, xdim)).T
        info.plan.trafo()
        samples = info.plan.f.copy() * info.pulsefac
        expected = self._expected_delta_ft(asymmetric_uv, xpix, ypix,
                                           xdim, ydim, psize)
        np.testing.assert_allclose(samples, expected, rtol=tol, atol=tol)
