"""End-to-end imaging tests (roadmap task 1F.1).

Each test simulates a synthetic observation of a known source, runs the
full imaging pipeline (compute_init_state -> compute_objective ->
compute_objective_grad -> scipy.optimize) via Imager.make_image, and
asserts the reconstruction recovers the source within image-quality
thresholds.

To stress the imager (rather than a near-truth prior just confirming
the wiring works) the tests use an *independent* 80 μas Gaussian prior
(not a blurred version of the source) and the standard EHT data-term
mix (amp + closure phase + log closure amplitude); polarimetric runs
add pvis + m. niter=3 exercises the converge loop with inter-round
blurring.

Covers all three transform types (direct / fast / nfft) and both
Stokes-I and polarimetric (IP) imaging.
"""

import numpy as np
import pytest

import ehtim as eh

# Independent prior FWHM (μas). Larger than gauss_im's 50 μas truth, so the
# imager has to actually shrink + resharpen rather than just refine.
PRIOR_FWHM_UAS = 80

# Reconstruction quality thresholds. nxcorr varies across scipy / numpy /
# BLAS versions because L-BFGS-B trajectories diverge slightly with
# different floating-point implementations: locally (Python 3.11) this
# config produces nxcorr ~0.98; CI Python 3.10 + older scipy lands ~0.87.
# 0.80 is a sanity-check floor that still catches real wiring regressions.
STOKES_I_NXCORR_MIN = 0.80
FLUX_REL_TOL = 0.05


def _independent_prior(total_flux, fwhm_uas=PRIOR_FWHM_UAS):
    """Build an independent (not-derived-from-source) Gaussian prior."""
    p = eh.image.make_empty(32, 200 * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
    return p.add_gauss(total_flux,
                       (fwhm_uas * eh.RADPERUAS, fwhm_uas * eh.RADPERUAS,
                        0, 0, 0))


def _imager_kwargs_stokes_i(ttype):
    """Standard EHT data-term mix for Stokes-I imaging."""
    return dict(
        data_term={"amp": 100, "cphase": 100, "logcamp": 50},
        reg_term={"simple": 1, "tv": 10},
        ttype=ttype, pol="I", maxit=100,
    )


def _imager_kwargs_stokes_i_diag(ttype):
    """Stokes-I mix using diagonalized closure quantities.

    `cphase_diag` and `logcamp_diag` consume per-timestamp diagonalized
    closures rather than the redundant minimal-set; data + sigma + A are
    dtype=object ragged arrays. Exercises that the full imaging pipeline
    converges through these paths, not just the per-term chi^2 / gradient.
    """
    return dict(
        data_term={"amp": 100, "cphase_diag": 100, "logcamp_diag": 50},
        reg_term={"simple": 1, "tv": 10},
        ttype=ttype, pol="I", maxit=100,
    )


def _imager_kwargs_polarimetric(ttype):
    """Same Stokes-I terms + linear-pol data terms for IP imaging."""
    return dict(
        data_term={"amp": 100, "cphase": 100, "logcamp": 50,
                   "pvis": 100, "m": 50},
        reg_term={"simple": 1, "tv": 10, "hw": 1, "ptv": 1},
        ttype=ttype, pol="IP", transform=["log", "mcv"], maxit=100,
    )


class TestEndToEndReconstruction:
    """Gaussian source -> simulate -> reconstruct -> verify (1F.1)."""

    @pytest.mark.parametrize("ttype", ["direct", "fast"])
    def test_recovers_gaussian_stokes_i(self, gauss_im, observe, ttype):
        obs = observe(gauss_im, ttype=ttype, seed=42)
        prior = _independent_prior(gauss_im.total_flux())
        imgr = eh.imager.Imager(
            obs, prior, prior_im=prior, flux=gauss_im.total_flux(),
            **_imager_kwargs_stokes_i(ttype),
        )
        out = imgr.make_image_I(niter=3, show_updates=False)

        assert out.total_flux() == pytest.approx(
            gauss_im.total_flux(), rel=FLUX_REL_TOL,
        )
        (errors, _, _) = out.compare_images(
            gauss_im, metric=["nxcorr"], blur_frac=0.0,
        )
        assert errors[0] > STOKES_I_NXCORR_MIN, (
            f"nxcorr={errors[0]:.3f} below {STOKES_I_NXCORR_MIN} threshold")

    def test_recovers_gaussian_stokes_i_nfft(self, gauss_im, observe):
        obs = observe(gauss_im, ttype="nfft", seed=42)
        prior = _independent_prior(gauss_im.total_flux())
        imgr = eh.imager.Imager(
            obs, prior, prior_im=prior, flux=gauss_im.total_flux(),
            **_imager_kwargs_stokes_i("nfft"),
        )
        out = imgr.make_image_I(niter=3, show_updates=False)

        assert out.total_flux() == pytest.approx(
            gauss_im.total_flux(), rel=FLUX_REL_TOL,
        )
        (errors, _, _) = out.compare_images(
            gauss_im, metric=["nxcorr"], blur_frac=0.0,
        )
        assert errors[0] > STOKES_I_NXCORR_MIN

    @pytest.mark.parametrize("ttype", ["direct", "fast"])
    def test_recovers_gaussian_stokes_i_diag(self, gauss_im, observe, ttype):
        obs = observe(gauss_im, ttype=ttype, seed=42)
        prior = _independent_prior(gauss_im.total_flux())
        imgr = eh.imager.Imager(
            obs, prior, prior_im=prior, flux=gauss_im.total_flux(),
            **_imager_kwargs_stokes_i_diag(ttype),
        )
        out = imgr.make_image_I(niter=3, show_updates=False)

        assert out.total_flux() == pytest.approx(
            gauss_im.total_flux(), rel=FLUX_REL_TOL,
        )
        (errors, _, _) = out.compare_images(
            gauss_im, metric=["nxcorr"], blur_frac=0.0,
        )
        assert errors[0] > STOKES_I_NXCORR_MIN, (
            f"diag nxcorr={errors[0]:.3f} below {STOKES_I_NXCORR_MIN} threshold")

    def test_recovers_gaussian_stokes_i_diag_nfft(self, gauss_im, observe):
        obs = observe(gauss_im, ttype="nfft", seed=42)
        prior = _independent_prior(gauss_im.total_flux())
        imgr = eh.imager.Imager(
            obs, prior, prior_im=prior, flux=gauss_im.total_flux(),
            **_imager_kwargs_stokes_i_diag("nfft"),
        )
        out = imgr.make_image_I(niter=3, show_updates=False)

        assert out.total_flux() == pytest.approx(
            gauss_im.total_flux(), rel=FLUX_REL_TOL,
        )
        (errors, _, _) = out.compare_images(
            gauss_im, metric=["nxcorr"], blur_frac=0.0,
        )
        assert errors[0] > STOKES_I_NXCORR_MIN

    def test_recovers_gaussian_polarimetric_ip(self, gauss_im_pol, observe):
        """Simultaneous Stokes I + P imaging on a polarized Gaussian source.

        Exercises the polarimetric path through compute_init_state
        (randompol_lin handling, mcv transform, IP which_solve layout).
        The prior is unpolarized, so the imager picks up randompol_lin
        initialization inside compute_init_state.
        """
        obs = observe(gauss_im_pol, ttype="direct", seed=42)
        # Unpolarized independent prior -> triggers randompol_lin init.
        prior = _independent_prior(gauss_im_pol.total_flux())
        imgr = eh.imager.Imager(
            obs, prior, prior_im=prior, flux=gauss_im_pol.total_flux(),
            **_imager_kwargs_polarimetric("direct"),
        )
        imgr.prior_next = imgr.prior_next.switch_polrep(
            polrep_out="stokes", pol_prim_out="I")
        imgr.init_next = imgr.init_next.switch_polrep(
            polrep_out="stokes", pol_prim_out="I")
        np.random.seed(0)
        out = imgr.make_image_IP(niter=3, show_updates=False)

        assert out.total_flux() == pytest.approx(
            gauss_im_pol.total_flux(), rel=FLUX_REL_TOL,
        )
        (errors_i, _, _) = out.compare_images(
            gauss_im_pol, pol="I", metric=["nxcorr"], blur_frac=0.0,
        )
        assert errors_i[0] > STOKES_I_NXCORR_MIN, (
            f"Stokes-I nxcorr={errors_i[0]:.3f} below {STOKES_I_NXCORR_MIN} threshold")

        # Polarimetric output should be non-trivial. Tighter pol-quality
        # thresholds depend on hyperparameter tuning; here we just confirm
        # the pipeline produced meaningful Q/U with the right magnitude.
        assert np.any(out.qvec != 0), "no Q recovered"
        assert np.any(out.uvec != 0), "no U recovered"
        p_total = np.sum(np.sqrt(out.qvec**2 + out.uvec**2)) * out.psize**2
        p_truth = np.sum(np.sqrt(gauss_im_pol.qvec**2 + gauss_im_pol.uvec**2)) * gauss_im_pol.psize**2
        assert p_total == pytest.approx(p_truth, rel=0.5), (
            f"total polarized flux {p_total:.3e} off from truth {p_truth:.3e}")


def test_imager_fast_ttype_warns_deprecated(gauss_im, observe):
    """Constructing an Imager with ttype='fast' emits a DeprecationWarning
    pointing to nfft (the FFT imaging gradients are inaccurate)."""
    obs = observe(gauss_im, ttype="nfft")
    prior = _independent_prior(gauss_im.total_flux())
    with pytest.warns(DeprecationWarning, match=r"ttype='fast'"):
        eh.imager.Imager(
            obs, prior, prior_im=prior, flux=gauss_im.total_flux(),
            **_imager_kwargs_stokes_i("fast"),
        )
