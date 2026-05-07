"""End-to-end imaging tests (roadmap task 1F.1).

Each test simulates a synthetic observation of a known source, runs the
full imaging pipeline (compute_init_state -> compute_objective ->
compute_objective_grad -> scipy.optimize) via Imager.make_image, and
asserts the reconstruction recovers the source within image-quality
thresholds. Covers all three transform types (direct / fast / nfft) and
both Stokes-I and polarimetric (IP) imaging.
"""

import importlib.util

import numpy as np
import pytest

import ehtim as eh


def _has_pynfft():
    return importlib.util.find_spec("pynfft") is not None


def _imager_kwargs_stokes_i(im, ttype):
    """Default L-BFGS-B-friendly hyperparameters for Stokes I recovery."""
    return dict(
        data_term={"vis": 100},
        reg_term={"simple": 1, "tv": 10},
        ttype=ttype, pol="I", maxit=100,
    )


def _imager_kwargs_polarimetric(im, ttype):
    """Defaults for simultaneous I + P imaging."""
    return dict(
        data_term={"vis": 100, "pvis": 100, "m": 50},
        reg_term={"simple": 1, "tv": 10, "hw": 1, "ptv": 1},
        ttype=ttype, pol="IP", transform=["log", "mcv"], maxit=100,
    )


class TestEndToEndReconstruction:
    """Gaussian source -> simulate -> reconstruct -> verify (1F.1)."""

    @pytest.mark.parametrize("ttype", ["direct", "fast"])
    def test_recovers_gaussian_stokes_i(self, gauss_im, observe, ttype):
        obs = observe(gauss_im, ttype=ttype, seed=42)
        prior = gauss_im.blur_circ(40 * eh.RADPERUAS)
        imgr = eh.imager.Imager(
            obs, prior, prior_im=prior, flux=gauss_im.total_flux(),
            **_imager_kwargs_stokes_i(gauss_im, ttype),
        )
        out = imgr.make_image_I(niter=1, show_updates=False)

        # total flux conserved within 5%
        assert out.total_flux() == pytest.approx(
            gauss_im.total_flux(), rel=0.05,
        )
        # image-domain similarity to source
        (errors, _, _) = out.compare_images(
            gauss_im, metric=["nxcorr"], blur_frac=0.0,
        )
        assert errors[0] > 0.85, f"nxcorr={errors[0]:.3f} below 0.85 threshold"

    @pytest.mark.skipif(not _has_pynfft(), reason="pyNFFT not installed")
    def test_recovers_gaussian_stokes_i_nfft(self, gauss_im, observe):
        obs = observe(gauss_im, ttype="nfft", seed=42)
        prior = gauss_im.blur_circ(40 * eh.RADPERUAS)
        imgr = eh.imager.Imager(
            obs, prior, prior_im=prior, flux=gauss_im.total_flux(),
            **_imager_kwargs_stokes_i(gauss_im, "nfft"),
        )
        out = imgr.make_image_I(niter=1, show_updates=False)

        assert out.total_flux() == pytest.approx(
            gauss_im.total_flux(), rel=0.05,
        )
        (errors, _, _) = out.compare_images(
            gauss_im, metric=["nxcorr"], blur_frac=0.0,
        )
        assert errors[0] > 0.85

    def test_recovers_gaussian_polarimetric_ip(self, gauss_im_pol, observe):
        """Simultaneous Stokes I + P imaging on a polarized Gaussian source.

        Exercises the polarimetric path through compute_init_state
        (randompol_lin handling, mcv transform, IP which_solve layout).
        """
        obs = observe(gauss_im_pol, ttype="direct", seed=42)
        prior = gauss_im_pol.blur_circ(40 * eh.RADPERUAS)
        imgr = eh.imager.Imager(
            obs, prior, prior_im=prior, flux=gauss_im_pol.total_flux(),
            **_imager_kwargs_polarimetric(gauss_im_pol, "direct"),
        )
        imgr.prior_next = imgr.prior_next.switch_polrep(
            polrep_out="stokes", pol_prim_out="I")
        imgr.init_next = imgr.init_next.switch_polrep(
            polrep_out="stokes", pol_prim_out="I")
        np.random.seed(0)
        out = imgr.make_image_IP(niter=1, show_updates=False)

        # Stokes I recovery
        assert out.total_flux() == pytest.approx(
            gauss_im_pol.total_flux(), rel=0.10,
        )
        (errors_i, _, _) = out.compare_images(
            gauss_im_pol, pol="I", metric=["nxcorr"], blur_frac=0.0,
        )
        assert errors_i[0] > 0.85, (
            f"Stokes-I nxcorr={errors_i[0]:.3f} below 0.85 threshold")

        # Sanity check: polarimetric output exists and is non-trivial.
        # Tighter pol-recovery thresholds depend on hyperparameter tuning;
        # here we just confirm the pipeline produced meaningful Q/U output.
        assert np.any(out.qvec != 0), "no Q recovered"
        assert np.any(out.uvec != 0), "no U recovered"
