"""Tests for pure functional backend in ehtim/imaging/imager_backend.py.

Each test verifies that a backend function produces identical output
to the corresponding Imager class method.
"""

import numpy as np
import pytest

import ehtim as eh
from ehtim.imaging.imager_backend import (
    compute_chisq_dict,
    compute_chisqgrad_dict,
    compute_embed,
    compute_reg_dict,
    compute_reggrad_dict,
    transform_gradients,
    transform_imarr,
)

# Parametrize over square, tall, and wide images
IMAGE_SHAPES = [
    (32, 32),  # square
    (32, 48),  # tall (ydim > xdim)
    (48, 32),  # wide (xdim > ydim)
    (31, 31),  # odd square
    (31, 33),  # odd rectangular
]

# Multifrequency test frequencies
REFFREQ_HZ = 230e9
MF_ALT_FREQ_HZ = 345e9

# Synthetic polarization fractions for polarimetric tests
POL_FRAC_Q = 0.1
POL_FRAC_U = 0.05

# Data terms covered by per-term parametrized chi^2 / grad tests.
# Diagonalized variants (cphase_diag, logcamp_diag) hit upstream ehtim bugs
# (NumPy 1.24+ inhomogeneous-array, deprecated `np.float`); tracked separately.
PER_TERM_DATATERMS = ["vis", "amp", "bs", "cphase", "camp", "logcamp"]

# Polarimetric data terms with required (pol, transform) per term.
PER_TERM_POL_CASES = [
    ("pvis", "IP", ["log", "mcv"]),
    ("m",    "IP", ["log", "mcv"]),
    ("vvis", "IV", ["log", "vcv"]),
]

PER_TERM_POL_GRAD_CASES = PER_TERM_POL_CASES

# Noisy-truth bounds: snrcut=3 keeps the linearized error-propagation valid
# for closure quantities; the (lo, hi) range is empirical over 20 seeds.
NOISY_SNRCUT = 3.0
NOISY_CHISQ_LO = 0.3
NOISY_CHISQ_HI = 10.0


class TestComputeEmbed:
    """Tests for compute_embed (extracted from Imager.set_embed)."""

    def test_basic_gaussian(self, gauss_im):
        """All-positive Gaussian image gives full mask."""
        im = gauss_im
        embed_mask, coord_matrix = compute_embed(
            im.imvec, im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        assert embed_mask.shape == (im.xdim * im.ydim,)
        assert np.all(embed_mask)
        assert coord_matrix.shape == (im.xdim * im.ydim, 2)

    def test_clipping(self, gauss_im):
        """High clipfloor masks out low-intensity pixels."""
        im = gauss_im
        clipfloor = np.median(im.imvec)
        embed_mask, coord_matrix = compute_embed(
            im.imvec, im.xdim, im.ydim, im.psize, clipfloor=clipfloor,
        )
        n_embed = np.sum(embed_mask)
        assert n_embed < im.xdim * im.ydim
        assert n_embed > 0
        assert coord_matrix.shape == (n_embed, 2)

    def test_clipfloor_too_large(self, gauss_im):
        """Clipfloor above max pixel raises exception."""
        im = gauss_im
        with pytest.raises(Exception, match="clipfloor too large"):
            compute_embed(
                im.imvec, im.xdim, im.ydim, im.psize,
                clipfloor=np.max(im.imvec) + 1.0,
            )

    def test_matches_imager(self, gauss_im, observe):
        """Backend compute_embed matches Imager.set_embed exactly."""
        obs = observe(gauss_im)
        imgr = eh.imager.Imager(obs, gauss_im, gauss_im, gauss_im.total_flux(),
                                ttype="direct")

        # Call the Imager method
        imgr.set_embed()

        # Call the backend function with the same args
        embed_mask, coord_matrix = compute_embed(
            imgr.prior_next.imvec, imgr.prior_next.xdim,
            imgr.prior_next.ydim, imgr.prior_next.psize,
            imgr.clipfloor_next,
        )

        np.testing.assert_array_equal(embed_mask, imgr._embed_mask)
        np.testing.assert_array_equal(coord_matrix, imgr._coord_matrix)

    @pytest.mark.parametrize("xdim,ydim", IMAGE_SHAPES)
    def test_shapes(self, make_rect_image, xdim, ydim):
        """Backend handles square, rectangular, and odd-dim images correctly."""
        im = make_rect_image(xdim, ydim)
        embed_mask, coord_matrix = compute_embed(
            im.imvec, im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        assert embed_mask.shape == (xdim * ydim,)
        assert coord_matrix.shape == (xdim * ydim, 2)

    def test_coord_matrix_units(self, gauss_im):
        """Coordinate matrix values are in radians (scaled by psize)."""
        im = gauss_im
        _, coord_matrix = compute_embed(
            im.imvec, im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        # Coordinates should be within +/- (xdim/2) * psize
        max_coord = (im.xdim // 2) * im.psize
        assert np.all(np.abs(coord_matrix) <= max_coord + im.psize)


# ---------------------------------------------------------------------------
# Helpers for chisq dict tests
# ---------------------------------------------------------------------------


def _call_backend_chisq_dict(imgr, imcur):
    """Call compute_chisq_dict with args pulled from an initialized Imager."""
    return compute_chisq_dict(
        imcur, sorted(imgr.dat_term_next.keys()), imgr._data_tuples,
        imgr.obslist_next, imgr._logfreqratio_list, imgr.mf_next,
        imgr.pol_next, imgr._ttype, imgr._embed_mask,
    )


def _call_backend_chisqgrad_dict(imgr, imcur):
    """Call compute_chisqgrad_dict with args pulled from an initialized Imager."""
    return compute_chisqgrad_dict(
        imcur, sorted(imgr.dat_term_next.keys()), imgr._data_tuples,
        imgr.obslist_next, imgr._logfreqratio_list, imgr.mf_next,
        imgr.pol_next, imgr._ttype, imgr._embed_mask,
        imgr._which_solve, imgr._nimage,
    )


def _call_backend_reg_dict(imgr, imcur, mf_flux=None):
    """Call compute_reg_dict with args pulled from an initialized Imager.

    `mf_flux` overrides imgr.mf_flux for tests that exercise the
    REGULARIZERS_ALLFREQS_I validation path.
    """
    return compute_reg_dict(
        imcur, sorted(imgr.reg_term_next.keys()), imgr._xprior, imgr._embed_mask,
        imgr.flux_next, imgr.pflux_next, imgr.vflux_next,
        imgr.prior_next.xdim, imgr.prior_next.ydim, imgr.prior_next.psize,
        imgr.norm_reg, imgr.beam_size, imgr.regparams,
        imgr.mf_next, imgr.mf_flux if mf_flux is None else mf_flux,
        imgr.obslist_next, imgr._logfreqratio_list, imgr.pol_next,
    )


def _call_backend_reggrad_dict(imgr, imcur, mf_flux=None):
    """Call compute_reggrad_dict with args pulled from an initialized Imager."""
    return compute_reggrad_dict(
        imcur, sorted(imgr.reg_term_next.keys()), imgr._xprior, imgr._embed_mask,
        imgr.flux_next, imgr.pflux_next, imgr.vflux_next,
        imgr.prior_next.xdim, imgr.prior_next.ydim, imgr.prior_next.psize,
        imgr.norm_reg, imgr.beam_size, imgr.regparams,
        imgr.mf_next, imgr.mf_flux if mf_flux is None else mf_flux,
        imgr.obslist_next, imgr._logfreqratio_list, imgr.pol_next,
        imgr._which_solve, imgr._nimage,
    )


class TestComputeChisqDict:
    """Tests for compute_chisq_dict (extracted from Imager.make_chisq_dict)."""

    @pytest.mark.parametrize("dterm", PER_TERM_DATATERMS)
    def test_chisq_zero_on_truth_no_debias(self, gauss_im, observe,
                                            initialize_imager, dterm):
        """chi^2 on the truth image is ~0 for every data term (noise-free, no debias).

        Tests forward-model correctness: predicted chi^2 == data when imcur is
        the truth image and we disable both the noise and debias corrections
        that would otherwise shift the data away from the raw truth values.
        """
        obs = observe(gauss_im)
        imgr, imcur = initialize_imager(obs, gauss_im, {dterm: 1}, debias=False)
        result = _call_backend_chisq_dict(imgr, imcur)
        assert set(result.keys()) == {dterm}
        assert result[dterm] < 1e-10

    @pytest.mark.parametrize("dterm", PER_TERM_DATATERMS)
    def test_chisq_near_unity_on_noisy_truth(self, gauss_im, observe,
                                              initialize_imager, dterm):
        """chi^2 on noisy truth is ~1 for every data term (with snrcut)."""
        obs = observe(gauss_im, seed=42)
        imgr, imcur = initialize_imager(obs, gauss_im, {dterm: 1},
                                        snrcut=NOISY_SNRCUT)
        result = _call_backend_chisq_dict(imgr, imcur)
        assert set(result.keys()) == {dterm}
        assert NOISY_CHISQ_LO < result[dterm] < NOISY_CHISQ_HI

    @pytest.mark.parametrize("dterm,pol,transform", PER_TERM_POL_CASES)
    def test_chisq_zero_on_truth_no_debias_pol(self, gauss_im_pol, observe,
                                                initialize_imager,
                                                dterm, pol, transform):
        """chi^2 on a polarized truth image is ~0 (noise-free, no debias)."""
        obs = observe(gauss_im_pol)
        imgr, imcur = initialize_imager(
            obs, gauss_im_pol, {dterm: 1},
            pol=pol, transform=transform, debias=False,
        )
        result = _call_backend_chisq_dict(imgr, imcur)
        assert set(result.keys()) == {dterm}
        assert result[dterm] < 1e-10

    @pytest.mark.parametrize("dterm,pol,transform", PER_TERM_POL_CASES)
    def test_chisq_near_unity_on_noisy_truth_pol(self, gauss_im_pol, observe,
                                                  initialize_imager,
                                                  dterm, pol, transform):
        """chi^2 on noisy polarized truth is ~1 for every pol data term."""
        obs = observe(gauss_im_pol, seed=42)
        imgr, imcur = initialize_imager(
            obs, gauss_im_pol, {dterm: 1},
            pol=pol, transform=transform, snrcut=NOISY_SNRCUT,
        )
        result = _call_backend_chisq_dict(imgr, imcur)
        assert set(result.keys()) == {dterm}
        assert NOISY_CHISQ_LO < result[dterm] < NOISY_CHISQ_HI

    def test_multiple_dataterms(self, gauss_im, observe, initialize_imager):
        """Multiple data terms — one entry per term, all finite."""
        obs = observe(gauss_im)
        data_term = {"vis": 100, "amp": 10, "cphase": 5}
        imgr, imcur = initialize_imager(obs, gauss_im, data_term)
        result = _call_backend_chisq_dict(imgr, imcur)
        assert set(result.keys()) == set(data_term.keys())
        for v in result.values():
            assert np.isfinite(v)

    def test_matches_imager(self, gauss_im, observe, initialize_imager):
        """Backend output == Imager.make_chisq_dict output (wrapper sanity check)."""
        obs = observe(gauss_im)
        imgr, imcur = initialize_imager(obs, gauss_im, {"vis": 100, "cphase": 10})

        method_result = imgr.make_chisq_dict(imcur)
        backend_result = _call_backend_chisq_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys()
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])

    def test_multiple_observations(self, gauss_im, observe, initialize_imager):
        """Two observations — keys become dname_i, each entry finite."""
        # Split the standard 0-24h observation window in two.
        obs1 = observe(gauss_im, tstop=12.0)
        obs2 = observe(gauss_im, tstart=12.0)
        imgr, imcur = initialize_imager(
            [obs1, obs2], gauss_im, {"vis": 100},
        )
        result = _call_backend_chisq_dict(imgr, imcur)
        assert set(result.keys()) == {"vis_0", "vis_1"}
        for v in result.values():
            assert np.isfinite(v)

    def test_multifrequency(self, gauss_im, observe, initialize_imager):
        """Multifrequency imaging — exercises the mf_next=True branch (image_at_freq)."""
        # Two observations at different frequencies, same source.
        im_lo = gauss_im.copy()
        im_lo.rf = REFFREQ_HZ
        im_hi = gauss_im.copy()
        im_hi.rf = MF_ALT_FREQ_HZ
        obs_lo = observe(im_lo)
        obs_hi = observe(im_hi)

        # Use im_lo (at reference freq) as the prior.
        imgr, imcur = initialize_imager(
            [obs_lo, obs_hi], im_lo, {"vis": 100}, mf=True, mf_order=1,
        )
        assert imgr.mf_next is True
        assert len(imgr._logfreqratio_list) == 2

        result = _call_backend_chisq_dict(imgr, imcur)
        assert set(result.keys()) == {"vis_0", "vis_1"}
        for v in result.values():
            assert np.isfinite(v)

    def test_polarimetric_stokes_i_bundled(self, gauss_im, observe, initialize_imager):
        """pol='IP' with DATATERMS entry: exercises the imcur_nu_I bug fix.

        Prior to the fix, imcur has shape (4, N) and was passed directly to
        imutils.chisq which would crash with a shape mismatch. After the fix
        the function slices imcur[0] for single-polarization data terms.
        """
        # Build a polarized image: Q and U scaled from Stokes I at a fixed angle.
        im = gauss_im.copy()
        qimage = POL_FRAC_Q * im.imarr()
        uimage = POL_FRAC_U * im.imarr()
        im.add_qu(qimage, uimage)

        obs = observe(im)
        imgr, imcur = initialize_imager(
            obs, im, {"vis": 100, "pvis": 100}, pol="IP",
        )
        # The buggy version crashes here with ValueError before returning.
        backend_result = _call_backend_chisq_dict(imgr, imcur)
        assert set(backend_result.keys()) == {"vis", "pvis"}
        for v in backend_result.values():
            assert np.isfinite(v)

        # Verify the wrapper (imgr.make_chisq_dict) also succeeds and matches.
        method_result = imgr.make_chisq_dict(imcur)
        assert method_result.keys() == backend_result.keys()
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])

    @pytest.mark.parametrize("ttype", ["direct", "fast", "nfft"])
    def test_all_ttypes(self, gauss_im, observe, initialize_imager, ttype):
        """Backend works for all three transform types."""
        if ttype == "nfft":
            pytest.importorskip("pynfft")
        obs = observe(gauss_im, ttype=ttype)
        imgr, imcur = initialize_imager(obs, gauss_im, {"vis": 100}, ttype=ttype)
        result = _call_backend_chisq_dict(imgr, imcur)
        assert np.isfinite(result["vis"])


class TestComputeChisqgradDict:
    """Tests for compute_chisqgrad_dict (extracted from Imager.make_chisqgrad_dict)."""

    @pytest.mark.parametrize("dterm", PER_TERM_DATATERMS)
    def test_grad_zero_on_truth_no_debias(self, gauss_im, observe,
                                           initialize_imager, dterm):
        """Gradient on the truth image is ~0 for every data term (noise-free, no debias)."""
        obs = observe(gauss_im)
        imgr, imcur = initialize_imager(obs, gauss_im, {dterm: 1}, debias=False)
        result = _call_backend_chisqgrad_dict(imgr, imcur)
        assert set(result.keys()) == {dterm}
        assert result[dterm].shape == (imgr._nimage,)
        assert np.max(np.abs(result[dterm])) < 1e-10

    @pytest.mark.parametrize("dterm", PER_TERM_DATATERMS)
    def test_grad_finite_on_noisy_truth(self, gauss_im, observe,
                                         initialize_imager, dterm):
        """Gradient on noisy truth is finite for every data term."""
        obs = observe(gauss_im, seed=42)
        imgr, imcur = initialize_imager(obs, gauss_im, {dterm: 1},
                                        snrcut=NOISY_SNRCUT)
        result = _call_backend_chisqgrad_dict(imgr, imcur)
        assert set(result.keys()) == {dterm}
        assert result[dterm].shape == (imgr._nimage,)
        assert np.all(np.isfinite(result[dterm]))

    @pytest.mark.parametrize("dterm,pol,transform", PER_TERM_POL_GRAD_CASES)
    def test_grad_zero_on_truth_no_debias_pol(self, gauss_im_pol, observe,
                                               initialize_imager,
                                               dterm, pol, transform):
        """Gradient on polarized truth is ~0 (noise-free, no debias)."""
        obs = observe(gauss_im_pol)
        imgr, imcur = initialize_imager(
            obs, gauss_im_pol, {dterm: 1},
            pol=pol, transform=transform, debias=False,
        )
        result = _call_backend_chisqgrad_dict(imgr, imcur)
        assert set(result.keys()) == {dterm}
        assert np.max(np.abs(result[dterm])) < 1e-10

    @pytest.mark.parametrize("dterm,pol,transform", PER_TERM_POL_GRAD_CASES)
    def test_grad_finite_on_noisy_truth_pol(self, gauss_im_pol, observe,
                                             initialize_imager,
                                             dterm, pol, transform):
        """Gradient on noisy polarized truth is finite for every pol term."""
        obs = observe(gauss_im_pol, seed=42)
        imgr, imcur = initialize_imager(
            obs, gauss_im_pol, {dterm: 1},
            pol=pol, transform=transform, snrcut=NOISY_SNRCUT,
        )
        result = _call_backend_chisqgrad_dict(imgr, imcur)
        assert set(result.keys()) == {dterm}
        assert np.all(np.isfinite(result[dterm]))

    def test_multiple_dataterms(self, gauss_im, observe, initialize_imager):
        """Multiple gradient entries, all finite, correct shape."""
        obs = observe(gauss_im)
        data_term = {"vis": 100, "amp": 10, "cphase": 5}
        imgr, imcur = initialize_imager(obs, gauss_im, data_term)
        result = _call_backend_chisqgrad_dict(imgr, imcur)
        assert set(result.keys()) == set(data_term.keys())
        for v in result.values():
            assert v.shape == (imgr._nimage,)
            assert np.all(np.isfinite(v))

    def test_matches_imager(self, gauss_im, observe, initialize_imager):
        """Backend output == Imager.make_chisqgrad_dict output (tight tol)."""
        obs = observe(gauss_im)
        imgr, imcur = initialize_imager(obs, gauss_im, {"vis": 100, "cphase": 10})

        method_result = imgr.make_chisqgrad_dict(imcur)
        backend_result = _call_backend_chisqgrad_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys()
        for key in method_result:
            np.testing.assert_allclose(method_result[key], backend_result[key],
                                       rtol=1e-15, atol=0)

    def test_multiple_observations(self, gauss_im, observe, initialize_imager):
        """Two observations — one gradient entry per obs."""
        # Split the standard 0-24h observation window in two.
        obs1 = observe(gauss_im, tstop=12.0)
        obs2 = observe(gauss_im, tstart=12.0)
        imgr, imcur = initialize_imager(
            [obs1, obs2], gauss_im, {"vis": 100},
        )
        result = _call_backend_chisqgrad_dict(imgr, imcur)
        assert set(result.keys()) == {"vis_0", "vis_1"}
        for v in result.values():
            assert v.shape == (imgr._nimage,)
            assert np.all(np.isfinite(v))

    def test_multifrequency(self, gauss_im, observe, initialize_imager):
        """Multifrequency imaging — exercises mf_all_grads_chain branch."""
        im_lo = gauss_im.copy()
        im_lo.rf = REFFREQ_HZ
        im_hi = gauss_im.copy()
        im_hi.rf = MF_ALT_FREQ_HZ
        obs_lo = observe(im_lo)
        obs_hi = observe(im_hi)
        imgr, imcur = initialize_imager(
            [obs_lo, obs_hi], im_lo, {"vis": 100}, mf=True, mf_order=1,
        )
        assert imgr.mf_next is True

        result = _call_backend_chisqgrad_dict(imgr, imcur)
        assert set(result.keys()) == {"vis_0", "vis_1"}
        for v in result.values():
            assert np.all(np.isfinite(v))

    def test_polarimetric_stokes_i_bundled(self, gauss_im, observe, initialize_imager):
        """pol='IP' with DATATERMS entry: Stokes-I gradient is bundled (4, nimage)."""
        im = gauss_im.copy()
        qimage = POL_FRAC_Q * im.imarr()
        uimage = POL_FRAC_U * im.imarr()
        im.add_qu(qimage, uimage)

        obs = observe(im)
        imgr, imcur = initialize_imager(
            obs, im, {"vis": 100, "pvis": 100}, pol="IP",
        )
        backend_result = _call_backend_chisqgrad_dict(imgr, imcur)
        assert set(backend_result.keys()) == {"vis", "pvis"}
        # Stokes-I gradient is bundled: (chi2grad_I, 0, 0, 0)
        assert backend_result["vis"].shape == (4, imgr._nimage)
        assert np.all(np.isfinite(backend_result["vis"]))
        # Polarimetric gradient is also 4-component
        assert backend_result["pvis"].shape[0] == 4
        assert np.all(np.isfinite(backend_result["pvis"]))

        # Verify wrapper matches.
        method_result = imgr.make_chisqgrad_dict(imcur)
        assert method_result.keys() == backend_result.keys()
        for key in method_result:
            np.testing.assert_allclose(method_result[key], backend_result[key],
                                       rtol=1e-15, atol=0)

    @pytest.mark.parametrize("ttype", ["direct", "fast", "nfft"])
    def test_all_ttypes(self, gauss_im, observe, initialize_imager, ttype):
        """Backend works for all three transform types."""
        if ttype == "nfft":
            pytest.importorskip("pynfft")
        obs = observe(gauss_im, ttype=ttype)
        imgr, imcur = initialize_imager(obs, gauss_im, {"vis": 100}, ttype=ttype)
        result = _call_backend_chisqgrad_dict(imgr, imcur)
        assert np.all(np.isfinite(result["vis"]))


def test_chisq_and_chisqgrad_share_keys(gauss_im, observe, initialize_imager):
    """Cross-cutting invariant: chisq and chisqgrad dicts share the same key set."""
    obs = observe(gauss_im)
    imgr, imcur = initialize_imager(
        obs, gauss_im, {"vis": 100, "amp": 10, "cphase": 5},
    )
    chisq = _call_backend_chisq_dict(imgr, imcur)
    chisqgrad = _call_backend_chisqgrad_dict(imgr, imcur)
    assert set(chisq.keys()) == set(chisqgrad.keys())


class TestComputeRegDict:
    """Tests for compute_reg_dict (extracted from Imager.make_reg_dict)."""

    def test_matches_imager_stokes_i(self, gauss_im, observe, initialize_imager):
        """Single-freq, pol='I', multi-key reg_term — exercises sort + REGULARIZERS branch."""
        obs = observe(gauss_im)
        imgr, imcur = initialize_imager(
            obs, gauss_im, {"vis": 100},
            reg_term={"simple": 1, "tv": 10, "l1": 5},
        )
        method_result = imgr.make_reg_dict(imcur)
        backend_result = _call_backend_reg_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys() == {"simple", "tv", "l1"}
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])

    def test_matches_imager_polarimetric(self, gauss_im_pol, observe, initialize_imager):
        """Single-freq, pol='IP' with REGULARIZERS_POL — exercises polregularizer branch."""
        obs = observe(gauss_im_pol)
        imgr, imcur = initialize_imager(
            obs, gauss_im_pol, {"pvis": 100},
            reg_term={"hw": 1, "ptv": 1},
            pol="IP", transform=["log", "mcv"],
        )
        method_result = imgr.make_reg_dict(imcur)
        backend_result = _call_backend_reg_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys() == {"hw", "ptv"}
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])

    def test_matches_imager_stokes_i_pol_bundled(self, gauss_im_pol, observe,
                                                  initialize_imager):
        """pol='IP' with REGULARIZERS — exercises the imcur[0] slice branch."""
        obs = observe(gauss_im_pol)
        imgr, imcur = initialize_imager(
            obs, gauss_im_pol, {"vis": 100, "pvis": 100},
            reg_term={"simple": 1, "hw": 1},
            pol="IP", transform=["log", "mcv"],
        )
        method_result = imgr.make_reg_dict(imcur)
        backend_result = _call_backend_reg_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys() == {"simple", "hw"}
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])

    def test_matches_imager_multifrequency(self, gauss_im, observe, initialize_imager):
        """Multifrequency: hits mf+REGULARIZERS, mf+REGULARIZERS_ALLFREQS_I, mf+REGULARIZERS_SPECTRAL.

        TODO: REGULARIZERS_POLSPECTRAL (l2_alphap, l2_betap, l2_rm, l2_cm, etc.)
        require a length-10 imcur (mf + full pol). Add coverage once a polarimetric
        multifreq fixture exists.
        """
        im_lo = gauss_im.copy()
        im_lo.rf = REFFREQ_HZ
        im_hi = gauss_im.copy()
        im_hi.rf = MF_ALT_FREQ_HZ
        obs_lo = observe(im_lo)
        obs_hi = observe(im_hi)

        imgr, imcur = initialize_imager(
            [obs_lo, obs_hi], im_lo, {"vis": 100},
            reg_term={"simple": 1, "flux_mf": 1, "l2_alpha": 1},
            mf=True, mf_order=1,
            mf_flux=[im_lo.total_flux(), im_hi.total_flux()],
        )

        method_result = imgr.make_reg_dict(imcur)
        backend_result = _call_backend_reg_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys() == {"simple", "flux_mf", "l2_alpha"}
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])

    def test_unknown_regname_raises(self, gauss_im, observe, initialize_imager):
        """Unknown regularizer name — backend raises with the documented message."""
        obs = observe(gauss_im)
        imgr, imcur = initialize_imager(obs, gauss_im, {"vis": 100})
        # Bypass Imager.check_params by passing reg_term_keys directly.
        with pytest.raises(Exception, match="not recognized"):
            compute_reg_dict(
                imcur, ["not_a_regularizer"], imgr._xprior, imgr._embed_mask,
                imgr.flux_next, imgr.pflux_next, imgr.vflux_next,
                imgr.prior_next.xdim, imgr.prior_next.ydim, imgr.prior_next.psize,
                imgr.norm_reg, imgr.beam_size, imgr.regparams,
                imgr.mf_next, imgr.mf_flux, imgr.obslist_next,
                imgr._logfreqratio_list, imgr.pol_next,
            )

    def test_mf_flux_validation(self, gauss_im, observe, initialize_imager):
        """mf=True + flux_mf regularizer + scalar mf_flux → raises."""
        im_lo = gauss_im.copy()
        im_lo.rf = REFFREQ_HZ
        im_hi = gauss_im.copy()
        im_hi.rf = MF_ALT_FREQ_HZ
        obs_lo = observe(im_lo)
        obs_hi = observe(im_hi)

        imgr, imcur = initialize_imager(
            [obs_lo, obs_hi], im_lo, {"vis": 100},
            reg_term={"flux_mf": 1},
            mf=True, mf_order=1,
        )
        # Pass a scalar (not a list) as mf_flux — must raise.
        with pytest.raises(Exception, match="mf_flux must be a list"):
            _call_backend_reg_dict(imgr, imcur, mf_flux=2.0)

    @pytest.mark.parametrize("xdim,ydim", IMAGE_SHAPES)
    def test_rect_images(self, make_rect_image, observe, initialize_imager,
                          xdim, ydim):
        """Backend handles rectangular (xdim != ydim) images."""
        im = make_rect_image(xdim, ydim)
        obs = observe(im)
        imgr, imcur = initialize_imager(
            obs, im, {"vis": 100},
            reg_term={"simple": 1, "tv": 10},
        )
        method_result = imgr.make_reg_dict(imcur)
        backend_result = _call_backend_reg_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys()
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])


class TestComputeReggradDict:
    """Tests for compute_reggrad_dict (extracted from Imager.make_reggrad_dict)."""

    def test_matches_imager_stokes_i(self, gauss_im, observe, initialize_imager):
        """Single-freq, pol='I' — gradient shape (nimage,), bit-identical to method."""
        obs = observe(gauss_im)
        imgr, imcur = initialize_imager(
            obs, gauss_im, {"vis": 100},
            reg_term={"simple": 1, "tv": 10, "l1": 5},
        )
        method_result = imgr.make_reggrad_dict(imcur)
        backend_result = _call_backend_reggrad_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys() == {"simple", "tv", "l1"}
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])
            assert method_result[key].shape == (imgr._nimage,)

    def test_matches_imager_polarimetric(self, gauss_im_pol, observe, initialize_imager):
        """pol='IP' with REGULARIZERS_POL — exercises polregularizergrad branch."""
        obs = observe(gauss_im_pol)
        imgr, imcur = initialize_imager(
            obs, gauss_im_pol, {"pvis": 100},
            reg_term={"hw": 1, "ptv": 1},
            pol="IP", transform=["log", "mcv"],
        )
        method_result = imgr.make_reggrad_dict(imcur)
        backend_result = _call_backend_reggrad_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys() == {"hw", "ptv"}
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])

    def test_matches_imager_stokes_i_pol_bundled(self, gauss_im_pol, observe,
                                                  initialize_imager):
        """pol='IP' with REGULARIZERS — exercises the (4, nimage) bundling branch."""
        obs = observe(gauss_im_pol)
        imgr, imcur = initialize_imager(
            obs, gauss_im_pol, {"vis": 100, "pvis": 100},
            reg_term={"simple": 1, "hw": 1},
            pol="IP", transform=["log", "mcv"],
        )
        method_result = imgr.make_reggrad_dict(imcur)
        backend_result = _call_backend_reggrad_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys() == {"simple", "hw"}
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])
        # Stokes I regularizer 'simple' is bundled to (4, nimage) for pol imaging.
        assert backend_result["simple"].shape == (4, imgr._nimage)

    def test_matches_imager_multifrequency(self, gauss_im, observe, initialize_imager):
        """Multifrequency: hits mf+REGULARIZERS, mf+REGULARIZERS_ALLFREQS_I, mf+REGULARIZERS_SPECTRAL.

        TODO: REGULARIZERS_POLSPECTRAL gradient paths (l2_alphap, l2_betap, l2_rm,
        l2_cm) require length-10 imcur (mf + full pol). Add coverage once a
        polarimetric multifreq fixture exists.
        """
        im_lo = gauss_im.copy()
        im_lo.rf = REFFREQ_HZ
        im_hi = gauss_im.copy()
        im_hi.rf = MF_ALT_FREQ_HZ
        obs_lo = observe(im_lo)
        obs_hi = observe(im_hi)

        imgr, imcur = initialize_imager(
            [obs_lo, obs_hi], im_lo, {"vis": 100},
            reg_term={"simple": 1, "flux_mf": 1, "l2_alpha": 1},
            mf=True, mf_order=1,
            mf_flux=[im_lo.total_flux(), im_hi.total_flux()],
        )

        method_result = imgr.make_reggrad_dict(imcur)
        backend_result = _call_backend_reggrad_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys() == {"simple", "flux_mf", "l2_alpha"}
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])

    def test_unknown_regname_raises(self, gauss_im, observe, initialize_imager):
        """Unknown regularizer name — backend raises with the documented message."""
        obs = observe(gauss_im)
        imgr, imcur = initialize_imager(obs, gauss_im, {"vis": 100})
        with pytest.raises(Exception, match="not recognized"):
            compute_reggrad_dict(
                imcur, ["not_a_regularizer"], imgr._xprior, imgr._embed_mask,
                imgr.flux_next, imgr.pflux_next, imgr.vflux_next,
                imgr.prior_next.xdim, imgr.prior_next.ydim, imgr.prior_next.psize,
                imgr.norm_reg, imgr.beam_size, imgr.regparams,
                imgr.mf_next, imgr.mf_flux, imgr.obslist_next,
                imgr._logfreqratio_list, imgr.pol_next,
                imgr._which_solve, imgr._nimage,
            )

    def test_mf_flux_validation(self, gauss_im, observe, initialize_imager):
        """mf=True + flux_mf regularizer + scalar mf_flux → raises."""
        im_lo = gauss_im.copy()
        im_lo.rf = REFFREQ_HZ
        im_hi = gauss_im.copy()
        im_hi.rf = MF_ALT_FREQ_HZ
        obs_lo = observe(im_lo)
        obs_hi = observe(im_hi)

        imgr, imcur = initialize_imager(
            [obs_lo, obs_hi], im_lo, {"vis": 100},
            reg_term={"flux_mf": 1},
            mf=True, mf_order=1,
        )
        with pytest.raises(Exception, match="mf_flux must be a list"):
            _call_backend_reggrad_dict(imgr, imcur, mf_flux=2.0)

    @pytest.mark.parametrize("xdim,ydim", IMAGE_SHAPES)
    def test_rect_images(self, make_rect_image, observe, initialize_imager,
                          xdim, ydim):
        """Backend handles rectangular (xdim != ydim) images."""
        im = make_rect_image(xdim, ydim)
        obs = observe(im)
        imgr, imcur = initialize_imager(
            obs, im, {"vis": 100},
            reg_term={"simple": 1, "tv": 10},
        )
        method_result = imgr.make_reggrad_dict(imcur)
        backend_result = _call_backend_reggrad_dict(imgr, imcur)

        assert method_result.keys() == backend_result.keys()
        for key in method_result:
            np.testing.assert_array_equal(method_result[key], backend_result[key])


def test_reg_and_reggrad_share_keys(gauss_im, observe, initialize_imager):
    """Cross-cutting invariant: reg and reggrad dicts share the same key set."""
    obs = observe(gauss_im)
    imgr, imcur = initialize_imager(
        obs, gauss_im, {"vis": 100},
        reg_term={"simple": 1, "tv": 10, "l1": 5},
    )
    reg = _call_backend_reg_dict(imgr, imcur)
    reggrad = _call_backend_reggrad_dict(imgr, imcur)
    assert set(reg.keys()) == set(reggrad.keys())


# ---------------------------------------------------------------------------
# transform_gradients: chain rule correctness via finite differences
# ---------------------------------------------------------------------------

def _fd_grad_of_transform(imarr, transforms, which_solve, scalar_obj):
    """Centered FD gradient of (scalar_obj o transform_imarr) w.r.t. solver vars.

    Used to verify transform_gradients applies the chain rule correctly.
    """
    eps = 1e-6
    grad = np.zeros_like(imarr)
    for k in range(imarr.shape[0]):
        for i in range(imarr.shape[1]):
            ip = imarr.copy()
            ip[k, i] += eps
            im = imarr.copy()
            im[k, i] -= eps
            f_p = scalar_obj(transform_imarr(ip, transforms, which_solve))
            f_m = scalar_obj(transform_imarr(im, transforms, which_solve))
            grad[k, i] = (f_p - f_m) / (2 * eps)
    return grad


class TestTransformGradientsChainRule:
    """transform_gradients chain rule agrees with FD of transform_imarr."""

    @staticmethod
    def _scalar_obj(phys):
        """Arbitrary smooth scalar objective: weighted sum of physical components."""
        weights = np.array([0.7, 1.3, -0.5, 0.9])[:, None]
        return float(np.sum(weights * phys**2))

    @staticmethod
    def _solver_imarr(seed=0, n=8, v_pre=None, m_pre=None, chi_pre=None):
        """Build a (4, n) solver-variable array; pass a scalar to pin a row."""
        rng = np.random.default_rng(seed)
        log_I = rng.uniform(-1.0, 1.0, size=n)
        m_arr = rng.uniform(-2.0, 2.0, size=n) if m_pre is None else np.full(n, m_pre)
        chi_arr = rng.uniform(-0.5, 0.5, size=n) if chi_pre is None else np.full(n, chi_pre)
        v_arr = rng.uniform(-0.3, 0.3, size=n) if v_pre is None else np.full(n, v_pre)
        return np.array([log_I, m_arr, chi_arr, v_arr])

    @pytest.mark.parametrize(
        "transforms,which_solve,imarr_kwargs,check_rows",
        [
            # ---- Real-usage operating points ----
            # IP imaging: mcv with vfrac=0 (no V). Solve I, m_pre, chi.
            (["log", "mcv"], np.array([1, 1, 1, 0]), {"v_pre": 0.0}, [0, 1, 2]),
            # IV imaging: vcv with mfrac=0 (no Q/U). Solve I, v_pre.
            (["log", "vcv"], np.array([1, 0, 0, 1]), {"m_pre": 0.0}, [0, 3]),
            # IPV imaging: polcv with diagonal Jacobian. Solve I, rho_pre, psi_pre.
            (["log", "polcv"], np.array([1, 1, 0, 1]), {}, [0, 1, 3]),
            # ---- General regimes (cross-term coverage) ----
            # mcv at vfrac != 0: dpsi/dmprime cross-term is nonzero.
            (["log", "mcv"], np.array([1, 1, 0, 0]), {}, [0, 1]),
            # vcv at mfrac != 0: dpsi/dvprime cross-term is nonzero.
            (["log", "vcv"], np.array([1, 0, 0, 1]), {}, [0, 3]),
            # ---- log only sanity ----
            (["log"], np.array([1, 0, 0, 0]), {}, [0]),
            # ---- pure pol cv (no log) sanity ----
            (["mcv"], np.array([0, 1, 0, 0]), {"v_pre": 0.0}, [1]),
            (["vcv"], np.array([0, 0, 0, 1]), {"m_pre": 0.0}, [3]),
            (["polcv"], np.array([0, 1, 0, 1]), {}, [1, 3]),
        ],
        ids=[
            "IP_vfrac0", "IV_mfrac0", "IPV",
            "IP_general", "IV_general",
            "log_only",
            "mcv_only", "vcv_only", "polcv_only",
        ],
    )
    def test_chain_rule_matches_finite_difference(self, transforms, which_solve,
                                                    imarr_kwargs, check_rows):
        """Chain rule output must match centered FD of (obj o transform)."""
        imarr = self._solver_imarr(seed=0, **imarr_kwargs)

        phys = transform_imarr(imarr, transforms, which_solve)
        weights = np.array([0.7, 1.3, -0.5, 0.9])[:, None]
        grad_phys = 2.0 * weights * phys

        grad_solver = transform_gradients(grad_phys, imarr, transforms, which_solve)
        grad_fd = _fd_grad_of_transform(
            imarr, transforms, which_solve, self._scalar_obj,
        )

        for k in check_rows:
            np.testing.assert_allclose(
                grad_solver[k], grad_fd[k],
                rtol=1e-5, atol=1e-8,
                err_msg=f"chain rule mismatch on row {k} for transforms={transforms}",
            )

    def test_log_chain_preserved_under_mcv(self):
        """outarr[0] from log+mcv must equal exp(imarr[0]) * gradarr[0]."""
        imarr = self._solver_imarr(seed=1, v_pre=0.0)
        gradarr = np.ones_like(imarr)
        which_solve = np.array([1, 1, 0, 0])

        out = transform_gradients(gradarr, imarr, ["log", "mcv"], which_solve)
        np.testing.assert_allclose(out[0], np.exp(imarr[0]), rtol=1e-12)

    def test_single_pol_log_chain_1d(self):
        """nimage==1 branch: 1D imarr, outarr = exp(imarr) * gradarr."""
        rng = np.random.default_rng(42)
        imarr = rng.uniform(-1.0, 1.0, size=8)
        gradarr = rng.uniform(-1.0, 1.0, size=8)
        which_solve = np.array([1, 0, 0, 0])

        out = transform_gradients(gradarr, imarr, ["log"], which_solve)
        np.testing.assert_allclose(out, np.exp(imarr) * gradarr, rtol=1e-12)

        out_no_log = transform_gradients(gradarr, imarr, [], which_solve)
        np.testing.assert_allclose(out_no_log, gradarr, rtol=1e-12)

    def test_multifreq_stokes_i_log_chain(self):
        """nimage==3 branch: log applies to row 0 only; spectral coeffs pass through."""
        rng = np.random.default_rng(7)
        imarr = rng.uniform(-1.0, 1.0, size=(3, 6))
        gradarr = rng.uniform(-1.0, 1.0, size=(3, 6))
        which_solve = np.array([1, 1, 1, 0])

        out = transform_gradients(gradarr, imarr, ["log"], which_solve)
        np.testing.assert_allclose(out[0], np.exp(imarr[0]) * gradarr[0], rtol=1e-12)
        np.testing.assert_array_equal(out[1], gradarr[1])
        np.testing.assert_array_equal(out[2], gradarr[2])

    def test_mf_pol_shape_smoke(self):
        """nimage==10 branch (mf+pol): shape preserved, log applied to row 0, rows 4-9 pass through."""
        rng = np.random.default_rng(13)
        n = 6
        imarr = np.empty((10, n))
        imarr[0] = rng.uniform(-1.0, 1.0, size=n)
        imarr[1] = rng.uniform(-2.0, 2.0, size=n)
        imarr[2] = rng.uniform(-0.5, 0.5, size=n)
        imarr[3] = 0.0  # vfrac pinned for pol='IP' regime
        imarr[4:10] = rng.uniform(-0.3, 0.3, size=(6, n))
        gradarr = rng.uniform(-1.0, 1.0, size=(10, n))
        which_solve = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])

        out = transform_gradients(gradarr, imarr, ["log", "mcv"], which_solve)

        assert out.shape == imarr.shape
        np.testing.assert_allclose(out[0], np.exp(imarr[0]) * gradarr[0], rtol=1e-12)
        np.testing.assert_array_equal(out[4:10], gradarr[4:10])

