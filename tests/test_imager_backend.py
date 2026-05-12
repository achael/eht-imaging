"""Tests for pure functional backend in ehtim/imaging/imager_backend.py.

Each test verifies that a backend function produces identical output
to the corresponding Imager class method.
"""

import numpy as np
import pytest

import ehtim as eh
from ehtim.imaging.imager_backend import (
    ImagerInitState,
    _pol_solve_block,
    compute_chisq_dict,
    compute_chisq_term,
    compute_chisqdata_term,
    compute_chisqgrad_dict,
    compute_chisqgrad_term,
    compute_data_tuples,
    compute_embed,
    compute_init_state,
    compute_logfreqratios,
    compute_objective,
    compute_objective_grad,
    compute_reg_dict,
    compute_reggrad_dict,
    compute_which_solve,
    make_initarr,
    pack_imarr,
    transform_gradients,
    transform_imarr,
    transform_imarr_inverse,
    unpack_imarr,
    validate_limits,
    validate_params,
)
from ehtim.imaging.imager_utils import embed_imarr

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


class TestComputeLogfreqratios:
    """Tests for compute_logfreqratios (extracted from Imager.init_imager
    and Imager.obslist_next setter)."""

    def test_single_frequency_at_reference(self):
        """Single freq at reffreq -> [0.0]."""
        result = compute_logfreqratios([230e9], 230e9)
        assert result == [0.0]

    def test_known_three_frequency_case(self):
        """Three frequencies bracketing reffreq."""
        result = compute_logfreqratios([86e9, 230e9, 345e9], 230e9)
        np.testing.assert_allclose(
            result,
            [np.log(86e9 / 230e9), 0.0, np.log(345e9 / 230e9)],
        )

    def test_monotonic_in_nu(self):
        """Higher nu than reffreq -> positive ratio; lower -> negative."""
        result = compute_logfreqratios([100e9, 230e9, 500e9], 230e9)
        assert result[0] < 0.0
        assert result[1] == 0.0
        assert result[2] > 0.0

    def test_returns_list(self):
        """Return type is a list (not a numpy array)."""
        result = compute_logfreqratios([230e9, 345e9], 230e9)
        assert isinstance(result, list)

    def test_empty_freq_list(self):
        """Empty input -> empty output (degenerate but well-defined)."""
        assert compute_logfreqratios([], 230e9) == []

    def test_matches_imager_obslist_setter(self, gauss_im, observe):
        """Backend output matches obslist_next.setter computation."""
        obs = observe(gauss_im)
        imgr = eh.imager.Imager(
            obs, gauss_im, prior_im=gauss_im, flux=gauss_im.total_flux(),
            data_term={"vis": 1}, ttype="direct", pol="I",
        )
        expected = compute_logfreqratios(imgr.freq_list, imgr.reffreq)
        assert imgr._logfreqratio_list == expected


class TestComputeWhichSolve:
    """Tests for compute_which_solve (extracted from Imager.init_imager)."""

    # --- single-frequency ---

    def test_sf_stokes_i(self):
        np.testing.assert_array_equal(
            compute_which_solve("I", mf=False), np.array([1]),
        )

    def test_sf_polarimetric_p(self):
        np.testing.assert_array_equal(
            compute_which_solve("P", mf=False), np.array([0, 1, 1, 0]),
        )

    def test_sf_polarimetric_ip(self):
        np.testing.assert_array_equal(
            compute_which_solve("IP", mf=False), np.array([1, 1, 1, 0]),
        )

    def test_sf_polarimetric_iv(self):
        np.testing.assert_array_equal(
            compute_which_solve("IV", mf=False), np.array([1, 0, 0, 1]),
        )

    def test_sf_polarimetric_ipv(self):
        np.testing.assert_array_equal(
            compute_which_solve("IPV", mf=False), np.array([1, 1, 1, 1]),
        )

    def test_sf_polarimetric_v(self):
        np.testing.assert_array_equal(
            compute_which_solve("V", mf=False), np.array([0, 0, 0, 1]),
        )

    # --- multi-frequency Stokes I ---

    @pytest.mark.parametrize("mf_order,expected", [
        (0, [1, 0, 0]),
        (1, [1, 1, 0]),
        (2, [1, 1, 1]),
    ])
    def test_mf_stokes_i_order(self, mf_order, expected):
        np.testing.assert_array_equal(
            compute_which_solve("I", mf=True, mf_order=mf_order),
            np.array(expected),
        )

    # --- multi-frequency polarimetric ---

    def test_mf_polarimetric_minimal(self):
        """MF + IP + all spectral orders 0 + no RM/CM."""
        np.testing.assert_array_equal(
            compute_which_solve("IP", mf=True,
                                mf_order=0, mf_order_pol=0,
                                mf_rm=False, mf_cm=False),
            np.array([1, 1, 1, 0,  # I, rho, phi, psi
                      0, 0,         # alpha, beta
                      0, 0,         # alpha_p, beta_p
                      0, 0]),       # RM, CM
        )

    def test_mf_polarimetric_all_on(self):
        """MF + IP with everything enabled."""
        np.testing.assert_array_equal(
            compute_which_solve("IP", mf=True,
                                mf_order=2, mf_order_pol=2,
                                mf_rm=True, mf_cm=True),
            np.array([1, 1, 1, 0,
                      1, 1,
                      1, 1,
                      1, 1]),
        )

    def test_mf_polarimetric_rm_only(self):
        np.testing.assert_array_equal(
            compute_which_solve("IP", mf=True,
                                mf_order=1, mf_order_pol=1,
                                mf_rm=True, mf_cm=False),
            np.array([1, 1, 1, 0,
                      1, 0,
                      1, 0,
                      1, 0]),
        )

    # --- error cases ---

    def test_raises_on_invalid_mf_order(self):
        with pytest.raises(Exception, match="mf_order must be 0, 1, or 2"):
            compute_which_solve("I", mf=True, mf_order=3)

    def test_raises_on_invalid_mf_order_pol(self):
        with pytest.raises(Exception, match="mf_order_pol must be 0, 1, or 2"):
            compute_which_solve("IP", mf=True, mf_order=1, mf_order_pol=3)

    def test_raises_on_mf_pol_without_p(self):
        with pytest.raises(Exception, match="requires pol_next=P"):
            compute_which_solve("IV", mf=True, mf_order=1, mf_order_pol=1)

    def test_raises_on_mf_pol_with_v(self):
        # 'IPV' contains both 'P' and 'V'; the V check fires.
        with pytest.raises(Exception, match="Stokes V not yet implemented"):
            compute_which_solve("IPV", mf=True, mf_order=1, mf_order_pol=1)

    # --- parity with init_imager ---

    def test_matches_imager_init_imager_stokes_i(self, gauss_im, observe):
        obs = observe(gauss_im)
        imgr = eh.imager.Imager(
            obs, gauss_im, prior_im=gauss_im, flux=gauss_im.total_flux(),
            data_term={"vis": 1}, ttype="direct", pol="I",
        )
        imgr.check_params()
        imgr.check_limits()
        imgr.init_imager()
        np.testing.assert_array_equal(
            imgr._which_solve,
            compute_which_solve(imgr.pol_next, imgr.mf_next,
                                mf_order=imgr.mf_order,
                                mf_order_pol=imgr.mf_order_pol,
                                mf_rm=imgr.mf_rm, mf_cm=imgr.mf_cm),
        )

    def test_matches_imager_init_imager_polarimetric(self, gauss_im_pol, observe):
        obs = observe(gauss_im_pol)
        imgr = eh.imager.Imager(
            obs, gauss_im_pol, prior_im=gauss_im_pol,
            flux=gauss_im_pol.total_flux(),
            data_term={"pvis": 1}, reg_term={"hw": 1},
            ttype="direct", pol="IP", transform=["log", "mcv"],
        )
        imgr.prior_next = imgr.prior_next.switch_polrep(
            polrep_out="stokes", pol_prim_out="I")
        imgr.init_next = imgr.init_next.switch_polrep(
            polrep_out="stokes", pol_prim_out="I")
        imgr.check_params()
        imgr.check_limits()
        imgr.init_imager()
        np.testing.assert_array_equal(
            imgr._which_solve,
            compute_which_solve(imgr.pol_next, imgr.mf_next,
                                mf_order=imgr.mf_order,
                                mf_order_pol=imgr.mf_order_pol,
                                mf_rm=imgr.mf_rm, mf_cm=imgr.mf_cm),
        )


def _call_compute_data_tuples(imgr):
    """Call compute_data_tuples with the args init_imager would pass."""
    return compute_data_tuples(
        imgr.obslist_next, imgr.prior_next, imgr._embed_mask,
        sorted(imgr.dat_term_next.keys()), imgr.pol_next,
        imgr._ttype,
        imgr._full_data_weighting_params(),
        imgr._full_fft_params(),
    )


class TestComputeDataTuples:
    """Tests for compute_data_tuples (extracted from Imager.init_imager)."""

    def test_single_obs_single_term(self, gauss_im, observe, initialize_imager):
        imgr, _ = initialize_imager(observe(gauss_im), gauss_im, {"vis": 1})
        tuples = _call_compute_data_tuples(imgr)
        assert set(tuples.keys()) == {"vis"}
        assert len(tuples["vis"]) == 3

    def test_multi_obs_key_suffixing(self, gauss_im, observe, initialize_imager):
        obs1 = observe(gauss_im)
        obs2 = observe(gauss_im)
        imgr, _ = initialize_imager([obs1, obs2], gauss_im, {"vis": 1})
        tuples = _call_compute_data_tuples(imgr)
        assert set(tuples.keys()) == {"vis_0", "vis_1"}

    def test_multiple_terms(self, gauss_im, observe, initialize_imager):
        imgr, _ = initialize_imager(
            observe(gauss_im), gauss_im, {"vis": 1, "amp": 1, "cphase": 1},
        )
        tuples = _call_compute_data_tuples(imgr)
        assert set(tuples.keys()) == {"vis", "amp", "cphase"}

    def test_polarimetric_term(self, gauss_im_pol, observe, initialize_imager):
        imgr, _ = initialize_imager(
            observe(gauss_im_pol), gauss_im_pol,
            {"pvis": 1}, reg_term={"hw": 1}, pol="IP",
            transform=["log", "mcv"],
        )
        tuples = _call_compute_data_tuples(imgr)
        assert set(tuples.keys()) == {"pvis"}
        assert len(tuples["pvis"]) == 3

    def test_mixed_pol_and_unpol_terms(self, gauss_im_pol, observe,
                                       initialize_imager):
        """pol=IP with both pvis (pol) and vis (unpol) data terms."""
        imgr, _ = initialize_imager(
            observe(gauss_im_pol), gauss_im_pol,
            {"vis": 1, "pvis": 1}, reg_term={"hw": 1},
            pol="IP", transform=["log", "mcv"],
        )
        tuples = _call_compute_data_tuples(imgr)
        assert set(tuples.keys()) == {"vis", "pvis"}

    def test_unrecognized_term_raises(self, gauss_im, observe,
                                      initialize_imager):
        imgr, _ = initialize_imager(observe(gauss_im), gauss_im, {"vis": 1})
        bogus_weighting = {**imgr._full_data_weighting_params(),
                           'snrcut': {'bogus': 0.0}}
        with pytest.raises(Exception, match="not recognized"):
            compute_data_tuples(
                imgr.obslist_next, imgr.prior_next, imgr._embed_mask,
                ["bogus"], imgr.pol_next,
                imgr._ttype, bogus_weighting, imgr._full_fft_params(),
            )

    def test_matches_imager_init_imager(self, gauss_im, observe,
                                        initialize_imager):
        """Parity: backend output equals dict populated by init_imager."""
        imgr, _ = initialize_imager(
            observe(gauss_im), gauss_im, {"vis": 1, "amp": 1},
        )
        result = _call_compute_data_tuples(imgr)
        assert set(result.keys()) == set(imgr._data_tuples.keys())
        for key, expected in imgr._data_tuples.items():
            for arr_a, arr_b in zip(result[key], expected, strict=True):
                np.testing.assert_array_equal(arr_a, arr_b)

    def test_matches_imager_init_imager_polarimetric(self, gauss_im_pol,
                                                      observe,
                                                      initialize_imager):
        imgr, _ = initialize_imager(
            observe(gauss_im_pol), gauss_im_pol,
            {"pvis": 1}, reg_term={"hw": 1}, pol="IP",
            transform=["log", "mcv"],
        )
        result = _call_compute_data_tuples(imgr)
        assert set(result.keys()) == set(imgr._data_tuples.keys())
        for key, expected in imgr._data_tuples.items():
            for arr_a, arr_b in zip(result[key], expected, strict=True):
                np.testing.assert_array_equal(arr_a, arr_b)


def _call_compute_init_state(imgr):
    """Call compute_init_state with args pulled from an initialized Imager."""
    return compute_init_state(
        imgr.obslist_next, imgr.init_next, imgr.prior_next,
        imgr.freq_list, imgr.reffreq,
        imgr.pol_next, imgr.mf_next, imgr.transform_next,
        imgr.mf_order, imgr.mf_order_pol, imgr.mf_rm, imgr.mf_cm,
        imgr.norm_init, imgr.flux_next, imgr.clipfloor_next,
        sorted(imgr.dat_term_next.keys()), imgr._ttype,
        imgr._full_data_weighting_params(),
        imgr._full_fft_params(),
    )


def _assert_state_matches_imager(state, imgr):
    """Field-by-field parity: ImagerInitState vs Imager._* attributes."""
    np.testing.assert_array_equal(state.init_arr, imgr._init_arr)
    np.testing.assert_array_equal(state.init_vec, imgr._init_vec)
    np.testing.assert_array_equal(state.prior_arr, imgr._prior_arr)
    np.testing.assert_array_equal(state.embed_mask, imgr._embed_mask)
    np.testing.assert_array_equal(state.coord_matrix, imgr._coord_matrix)
    np.testing.assert_array_equal(state.which_solve, imgr._which_solve)
    assert state.logfreqratio_list == imgr._logfreqratio_list
    assert state.nimage == imgr._nimage
    assert state.reffreq == imgr.reffreq
    assert set(state.data_tuples.keys()) == set(imgr._data_tuples.keys())
    for key, expected in imgr._data_tuples.items():
        for arr_a, arr_b in zip(state.data_tuples[key], expected, strict=True):
            np.testing.assert_array_equal(arr_a, arr_b)


class TestComputeInitState:
    """Tests for compute_init_state (orchestrator extracted from
    Imager.init_imager). Parity tests run init_imager first to populate
    Imager attributes, then call compute_init_state with the same args
    and assert field-by-field equality."""

    def test_returns_imager_init_state(self, gauss_im, observe,
                                       initialize_imager):
        imgr, _ = initialize_imager(observe(gauss_im), gauss_im, {"vis": 1})
        state = _call_compute_init_state(imgr)
        assert isinstance(state, ImagerInitState)

    def test_matches_imager_init_imager_stokes_i(self, gauss_im, observe,
                                                  initialize_imager):
        imgr, _ = initialize_imager(
            observe(gauss_im), gauss_im, {"vis": 1, "amp": 1},
            reg_term={"simple": 1, "tv": 10},
        )
        np.random.seed(0)
        state = _call_compute_init_state(imgr)
        _assert_state_matches_imager(state, imgr)

    def test_matches_imager_init_imager_polarimetric_ip(self, gauss_im_pol,
                                                         observe,
                                                         initialize_imager):
        np.random.seed(0)
        imgr, _ = initialize_imager(
            observe(gauss_im_pol), gauss_im_pol,
            {"pvis": 1}, reg_term={"hw": 1}, pol="IP",
            transform=["log", "mcv"],
        )
        np.random.seed(0)
        state = _call_compute_init_state(imgr)
        _assert_state_matches_imager(state, imgr)

    def test_matches_imager_init_imager_polarimetric_iv(self, gauss_im_pol,
                                                         observe,
                                                         initialize_imager):
        np.random.seed(0)
        imgr, _ = initialize_imager(
            observe(gauss_im_pol), gauss_im_pol,
            {"vvis": 1}, reg_term={"l2v": 1}, pol="IV",
            transform=["log", "vcv"],
        )
        np.random.seed(0)
        state = _call_compute_init_state(imgr)
        _assert_state_matches_imager(state, imgr)

    def test_matches_imager_init_imager_multi_obs(self, gauss_im, observe,
                                                   initialize_imager):
        obs1 = observe(gauss_im)
        obs2 = observe(gauss_im)
        imgr, _ = initialize_imager([obs1, obs2], gauss_im, {"vis": 1})
        state = _call_compute_init_state(imgr)
        _assert_state_matches_imager(state, imgr)

    @pytest.mark.parametrize("xdim,ydim", IMAGE_SHAPES)
    def test_rect_image_shapes(self, make_rect_image, eht_array, observe,
                                initialize_imager, xdim, ydim):
        im = make_rect_image(xdim, ydim)
        imgr, _ = initialize_imager(observe(im), im, {"vis": 1})
        state = _call_compute_init_state(imgr)
        _assert_state_matches_imager(state, imgr)

    def test_compute_data_false_uses_prior_tuples(self, gauss_im, observe,
                                                   initialize_imager):
        imgr, _ = initialize_imager(observe(gauss_im), gauss_im, {"vis": 1})
        sentinel = {"vis": ("dummy", "tuple", "sentinel")}
        state = compute_init_state(
            imgr.obslist_next, imgr.init_next, imgr.prior_next,
            imgr.freq_list, imgr.reffreq,
            imgr.pol_next, imgr.mf_next, imgr.transform_next,
            imgr.mf_order, imgr.mf_order_pol, imgr.mf_rm, imgr.mf_cm,
            imgr.norm_init, imgr.flux_next, imgr.clipfloor_next,
            sorted(imgr.dat_term_next.keys()), imgr._ttype,
            imgr._full_data_weighting_params(),
            imgr._full_fft_params(),
            compute_data=False, prior_data_tuples=sentinel,
        )
        assert state.data_tuples is sentinel

    def test_raises_on_obslist_freq_mismatch(self, gauss_im, observe,
                                              initialize_imager):
        imgr, _ = initialize_imager(observe(gauss_im), gauss_im, {"vis": 1})
        with pytest.raises(Exception, match="len\\(obslist\\) and len\\(freq_list\\)"):
            compute_init_state(
                imgr.obslist_next, imgr.init_next, imgr.prior_next,
                imgr.freq_list + [230e9],  # length mismatch
                imgr.reffreq,
                imgr.pol_next, imgr.mf_next, imgr.transform_next,
                imgr.mf_order, imgr.mf_order_pol, imgr.mf_rm, imgr.mf_cm,
                imgr.norm_init, imgr.flux_next, imgr.clipfloor_next,
                sorted(imgr.dat_term_next.keys()), imgr._ttype,
                imgr._full_data_weighting_params(),
                imgr._full_fft_params(),
            )


# ---------------------------------------------------------------------------
# Helpers for chisq dict tests
# ---------------------------------------------------------------------------


def _call_backend_chisq_dict(imgr, imcur):
    """Call compute_chisq_dict with args pulled from an initialized Imager."""
    return compute_chisq_dict(
        imcur, sorted(imgr.dat_term_next.keys()),
        imgr.mf_next, imgr.pol_next,
        imgr._data_tuples, imgr._logfreqratio_list, len(imgr.obslist_next),
        imgr._ttype, imgr._embed_mask,
    )


def _call_backend_chisqgrad_dict(imgr, imcur):
    """Call compute_chisqgrad_dict with args pulled from an initialized Imager."""
    return compute_chisqgrad_dict(
        imcur, sorted(imgr.dat_term_next.keys()),
        imgr.mf_next, imgr.pol_next,
        imgr._data_tuples, imgr._logfreqratio_list, len(imgr.obslist_next),
        imgr._ttype, imgr._embed_mask,
        imgr._which_solve, imgr._nimage,
    )


def _build_regparams(imgr, mf_flux=None):
    """Bundle all regularizer params from an Imager into a single dict.

    `mf_flux` overrides imgr.mf_flux for tests that exercise the
    REGULARIZERS_ALLFREQS_I validation path.
    """
    return {
        'flux': imgr.flux_next,
        'pflux': imgr.pflux_next,
        'vflux': imgr.vflux_next,
        'xdim': imgr.prior_next.xdim,
        'ydim': imgr.prior_next.ydim,
        'psize': imgr.prior_next.psize,
        'beam_size': imgr.beam_size,
        'mf_flux': imgr.mf_flux if mf_flux is None else mf_flux,
        **imgr.regparams,
    }


def _call_backend_reg_dict(imgr, imcur, mf_flux=None):
    """Call compute_reg_dict with args pulled from an initialized Imager."""
    return compute_reg_dict(
        imcur, sorted(imgr.reg_term_next.keys()),
        imgr.mf_next, imgr.pol_next,
        imgr._logfreqratio_list, len(imgr.obslist_next),
        imgr._prior_arr, imgr.norm_reg, _build_regparams(imgr, mf_flux=mf_flux),
        imgr._embed_mask,
    )


def _call_backend_reggrad_dict(imgr, imcur, mf_flux=None):
    """Call compute_reggrad_dict with args pulled from an initialized Imager."""
    return compute_reggrad_dict(
        imcur, sorted(imgr.reg_term_next.keys()),
        imgr.mf_next, imgr.pol_next,
        imgr._logfreqratio_list, len(imgr.obslist_next),
        imgr._prior_arr, imgr.norm_reg, _build_regparams(imgr, mf_flux=mf_flux),
        imgr._embed_mask,
        imgr._which_solve, imgr._nimage,
    )


def _call_backend_objective(imgr, imvec):
    """Call compute_objective with args pulled from an initialized Imager."""
    return compute_objective(
        imvec, imgr._init_arr,
        imgr.mf_next, imgr.pol_next,
        imgr._which_solve, imgr._data_tuples,
        imgr._logfreqratio_list, len(imgr.obslist_next),
        imgr.dat_term_next, imgr.reg_term_next,
        imgr._prior_arr, imgr.norm_reg, _build_regparams(imgr),
        imgr.transform_next, imgr._embed_mask, imgr._ttype,
    )


def _call_backend_objective_grad(imgr, imvec):
    """Call compute_objective_grad with args pulled from an initialized Imager."""
    return compute_objective_grad(
        imvec, imgr._init_arr,
        imgr.mf_next, imgr.pol_next,
        imgr._which_solve, imgr._data_tuples,
        imgr._logfreqratio_list, len(imgr.obslist_next),
        imgr.dat_term_next, imgr.reg_term_next,
        imgr._prior_arr, imgr.norm_reg, _build_regparams(imgr),
        imgr.transform_next, imgr._embed_mask, imgr._ttype, imgr._nimage,
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
                imcur, ["not_a_regularizer"],
                imgr.mf_next, imgr.pol_next,
                imgr._logfreqratio_list, len(imgr.obslist_next),
                imgr._prior_arr, imgr.norm_reg, _build_regparams(imgr),
                imgr._embed_mask,
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
                imcur, ["not_a_regularizer"],
                imgr.mf_next, imgr.pol_next,
                imgr._logfreqratio_list, len(imgr.obslist_next),
                imgr._prior_arr, imgr.norm_reg, _build_regparams(imgr),
                imgr._embed_mask,
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


class TestComputeObjective:
    """Tests for compute_objective (extracted from Imager.objfunc)."""

    def test_matches_imager_stokes_i(self, gauss_im, observe, initialize_imager):
        """Single-freq, pol='I', multi-key dat + reg — exercises sort + branch coverage."""
        obs = observe(gauss_im)
        imgr, _ = initialize_imager(
            obs, gauss_im, {"vis": 100, "amp": 10},
            reg_term={"simple": 1, "tv": 10},
        )
        imvec = imgr._init_vec
        backend_value = _call_backend_objective(imgr, imvec)
        method_value = imgr.objfunc(imvec)
        np.testing.assert_array_equal(backend_value, method_value)

    def test_matches_imager_polarimetric(self, gauss_im_pol, observe, initialize_imager):
        """pol='IP' with REGULARIZERS_POL — exercises polarimetric chisq + reg branches."""
        obs = observe(gauss_im_pol)
        imgr, _ = initialize_imager(
            obs, gauss_im_pol, {"pvis": 100},
            reg_term={"hw": 1, "ptv": 1},
            pol="IP", transform=["log", "mcv"],
        )
        imvec = imgr._init_vec
        backend_value = _call_backend_objective(imgr, imvec)
        method_value = imgr.objfunc(imvec)
        np.testing.assert_array_equal(backend_value, method_value)

    def test_matches_imager_stokes_i_pol_bundled(self, gauss_im_pol, observe,
                                                   initialize_imager):
        """pol='IP' with mixed Stokes-I + pol terms — both bundling branches active."""
        obs = observe(gauss_im_pol)
        imgr, _ = initialize_imager(
            obs, gauss_im_pol, {"vis": 100, "pvis": 100},
            reg_term={"simple": 1, "hw": 1},
            pol="IP", transform=["log", "mcv"],
        )
        imvec = imgr._init_vec
        backend_value = _call_backend_objective(imgr, imvec)
        method_value = imgr.objfunc(imvec)
        np.testing.assert_array_equal(backend_value, method_value)

    def test_matches_imager_multifrequency(self, gauss_im, observe, initialize_imager):
        """Two obs at different rf — hits the f'{dname}_{i}' key suffix branch."""
        im_lo = gauss_im.copy()
        im_lo.rf = REFFREQ_HZ
        im_hi = gauss_im.copy()
        im_hi.rf = MF_ALT_FREQ_HZ
        obs_lo = observe(im_lo)
        obs_hi = observe(im_hi)

        imgr, _ = initialize_imager(
            [obs_lo, obs_hi], im_lo, {"vis": 100},
            reg_term={"simple": 1, "flux_mf": 1, "l2_alpha": 1},
            mf=True, mf_order=1,
            mf_flux=[im_lo.total_flux(), im_hi.total_flux()],
        )
        imvec = imgr._init_vec
        backend_value = _call_backend_objective(imgr, imvec)
        method_value = imgr.objfunc(imvec)
        np.testing.assert_array_equal(backend_value, method_value)

    @pytest.mark.parametrize("xdim,ydim", IMAGE_SHAPES)
    def test_rect_images(self, make_rect_image, observe, initialize_imager,
                          xdim, ydim):
        """Backend handles rectangular (xdim != ydim) images end-to-end."""
        im = make_rect_image(xdim, ydim)
        obs = observe(im)
        imgr, _ = initialize_imager(
            obs, im, {"vis": 100},
            reg_term={"simple": 1, "tv": 10},
        )
        imvec = imgr._init_vec
        backend_value = _call_backend_objective(imgr, imvec)
        method_value = imgr.objfunc(imvec)
        np.testing.assert_array_equal(backend_value, method_value)


def _fd_grad_of_objective(imgr, imvec, indices, eps=1e-6):
    """Centered finite-difference gradient of compute_objective at given indices.

    Returns a full-length grad array but only the entries at `indices` are
    populated. Independent of compute_objective_grad — usable as ground truth
    for the analytic gradient under test.
    """
    grad = np.zeros_like(imvec)
    for i in indices:
        v_p = imvec.copy()
        v_p[i] += eps
        v_m = imvec.copy()
        v_m[i] -= eps
        f_p = _call_backend_objective(imgr, v_p)
        f_m = _call_backend_objective(imgr, v_m)
        grad[i] = (f_p - f_m) / (2 * eps)
    return grad


class TestComputeObjectiveGrad:
    """Tests for compute_objective_grad (extracted from Imager.objgrad)."""

    def test_matches_imager_stokes_i(self, gauss_im, observe, initialize_imager):
        """Single-freq, pol='I' — gradient bit-identical to method, 1D shape."""
        obs = observe(gauss_im)
        imgr, _ = initialize_imager(
            obs, gauss_im, {"vis": 100, "amp": 10},
            reg_term={"simple": 1, "tv": 10},
        )
        imvec = imgr._init_vec
        backend_grad = _call_backend_objective_grad(imgr, imvec)
        method_grad = imgr.objgrad(imvec)
        np.testing.assert_array_equal(backend_grad, method_grad)
        assert backend_grad.shape == imvec.shape

    def test_matches_imager_polarimetric(self, gauss_im_pol, observe, initialize_imager):
        """pol='IP' with REGULARIZERS_POL — exercises polchisqgrad + polreggrad chain."""
        obs = observe(gauss_im_pol)
        imgr, _ = initialize_imager(
            obs, gauss_im_pol, {"pvis": 100},
            reg_term={"hw": 1, "ptv": 1},
            pol="IP", transform=["log", "mcv"],
        )
        imvec = imgr._init_vec
        backend_grad = _call_backend_objective_grad(imgr, imvec)
        method_grad = imgr.objgrad(imvec)
        np.testing.assert_array_equal(backend_grad, method_grad)

    def test_matches_imager_stokes_i_pol_bundled(self, gauss_im_pol, observe,
                                                   initialize_imager):
        """pol='IP' with mixed Stokes-I + pol terms — cross-Stokes chain rule path."""
        obs = observe(gauss_im_pol)
        imgr, _ = initialize_imager(
            obs, gauss_im_pol, {"vis": 100, "pvis": 100},
            reg_term={"simple": 1, "hw": 1},
            pol="IP", transform=["log", "mcv"],
        )
        imvec = imgr._init_vec
        backend_grad = _call_backend_objective_grad(imgr, imvec)
        method_grad = imgr.objgrad(imvec)
        np.testing.assert_array_equal(backend_grad, method_grad)

    def test_matches_imager_multifrequency(self, gauss_im, observe, initialize_imager):
        """Two obs at different rf — multifreq grad bundling + key suffix branch."""
        im_lo = gauss_im.copy()
        im_lo.rf = REFFREQ_HZ
        im_hi = gauss_im.copy()
        im_hi.rf = MF_ALT_FREQ_HZ
        obs_lo = observe(im_lo)
        obs_hi = observe(im_hi)

        imgr, _ = initialize_imager(
            [obs_lo, obs_hi], im_lo, {"vis": 100},
            reg_term={"simple": 1, "flux_mf": 1, "l2_alpha": 1},
            mf=True, mf_order=1,
            mf_flux=[im_lo.total_flux(), im_hi.total_flux()],
        )
        imvec = imgr._init_vec
        backend_grad = _call_backend_objective_grad(imgr, imvec)
        method_grad = imgr.objgrad(imvec)
        np.testing.assert_array_equal(backend_grad, method_grad)

    def test_fd_matches_analytic_stokes_i(self, gauss_im, observe, initialize_imager):
        """FD of compute_objective ≈ compute_objective_grad on solved-for indices.

        Independent verification of the analytic gradient via centered FD.
        Samples 20 random indices to keep the test under a few seconds.
        """
        obs = observe(gauss_im)
        imgr, _ = initialize_imager(
            obs, gauss_im, {"vis": 100},
            reg_term={"simple": 1, "tv": 10},
        )
        imvec = imgr._init_vec

        rng = np.random.default_rng(42)
        indices = rng.choice(len(imvec), size=20, replace=False)

        grad_analytic = _call_backend_objective_grad(imgr, imvec)
        grad_fd = _fd_grad_of_objective(imgr, imvec, indices)

        np.testing.assert_allclose(
            grad_analytic[indices], grad_fd[indices],
            rtol=1e-4, atol=1e-8,
        )

    def test_fd_matches_analytic_polarimetric(self, gauss_im_pol, observe,
                                                initialize_imager):
        """FD verification under pol='IP' with log+mcv transforms.

        Exercises the chain rule applied through both the log (Stokes I) and
        mcv (polarization) transform components, on top of the polarimetric
        forward model. atol=1e-7 covers the FD noise floor: with chisq values
        ~O(1e6) and eps=1e-6, FD precision floors near 1e-7 in absolute terms.
        """
        obs = observe(gauss_im_pol)
        imgr, _ = initialize_imager(
            obs, gauss_im_pol, {"vis": 100, "pvis": 100},
            reg_term={"simple": 1, "hw": 1},
            pol="IP", transform=["log", "mcv"],
        )
        imvec = imgr._init_vec

        rng = np.random.default_rng(7)
        indices = rng.choice(len(imvec), size=20, replace=False)

        grad_analytic = _call_backend_objective_grad(imgr, imvec)
        grad_fd = _fd_grad_of_objective(imgr, imvec, indices)

        np.testing.assert_allclose(
            grad_analytic[indices], grad_fd[indices],
            rtol=1e-4, atol=1e-7,
        )

    @pytest.mark.parametrize("xdim,ydim", IMAGE_SHAPES)
    def test_rect_images(self, make_rect_image, observe, initialize_imager,
                          xdim, ydim):
        """Backend handles rectangular (xdim != ydim) images end-to-end."""
        im = make_rect_image(xdim, ydim)
        obs = observe(im)
        imgr, _ = initialize_imager(
            obs, im, {"vis": 100},
            reg_term={"simple": 1, "tv": 10},
        )
        imvec = imgr._init_vec
        backend_grad = _call_backend_objective_grad(imgr, imvec)
        method_grad = imgr.objgrad(imvec)
        np.testing.assert_array_equal(backend_grad, method_grad)


def test_objective_and_objective_grad_share_consistency(gauss_im, observe,
                                                          initialize_imager):
    """Cross-cutting invariant: gradient finite-norm and re-evaluation parity.

    Uses 'simple' regularizer only; TV/L1 can produce NaN at zero-pixel
    edges of the unblurred prior, which is unrelated to objective wiring.
    """
    obs = observe(gauss_im)
    imgr, _ = initialize_imager(
        obs, gauss_im, {"vis": 100, "amp": 10},
        reg_term={"simple": 1},
    )
    imvec = imgr._init_vec
    obj = _call_backend_objective(imgr, imvec)
    grad = _call_backend_objective_grad(imgr, imvec)
    assert np.isfinite(obj)
    assert np.all(np.isfinite(grad))
    assert grad.shape == imvec.shape
    # Re-evaluation parity through the method wrappers
    np.testing.assert_array_equal(obj, imgr.objfunc(imvec))
    np.testing.assert_array_equal(grad, imgr.objgrad(imvec))


# ---------------------------------------------------------------------------
# transform_gradients: chain rule correctness via finite differences
# ---------------------------------------------------------------------------

def _fd_grad_of_transform(imarr, transforms, which_solve, scalar_obj):
    """Centered finite-difference gradient of (scalar_obj o transform_imarr).

    Returns d/dimarr [scalar_obj(transform_imarr(imarr))] — the gradient of
    the composed function in solver-space. Computed numerically to second order
    via centered differences with eps=1e-6 (precision ~1e-10 in float64).
    Independent of transform_gradients itself, so suitable as ground truth
    for the chain rule it implements.
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
    """Verify transform_gradients implements the J^T-times-grad chain rule.

    `transform_gradients(grad_phys, imarr, transforms, which_solve)` converts
    a gradient w.r.t. PHYSICAL space (df/dphys, where phys = transform_imarr(imarr))
    into a gradient w.r.t. SOLVER space (df/dimarr), via the chain rule
    df/dimarr = J^T @ df/dphys, where J is the Jacobian of transform_imarr.

    Strategy: pick any smooth scalar f(phys), compute df/dimarr two ways —
    (1) the chain rule under test, and (2) centered finite differences applied
    to (f o transform_imarr). They must agree to FD precision (~1e-6).

    Coverage spans:
      - Real-usage operating points where ehtim pins one slot to enforce the
        diagonal-Jacobian regime (mcv ⇒ vfrac=0 for pol='IP'; vcv ⇒ mfrac=0
        for pol='IV'; polcv has a diagonal Jacobian everywhere).
      - General regimes where Jacobians have non-zero off-diagonal entries.
        These exercise the cross-derivative terms (drho/dmprime ↔ phys[3] for
        mcv, drho/dvprime ↔ phys[1] for vcv).
      - Pure transforms in isolation (no `log` composition).
      - Shape-dispatch branches: nimage==1 (single Stokes I 1D), nimage==3
        (multifreq Stokes I), nimage==4 (single-freq pol), nimage==10 (mf+pol).
    """

    @staticmethod
    def _scalar_obj(phys):
        """Arbitrary smooth scalar f(phys) = sum(weights * phys**2).

        Choice is deliberately generic — any C^2 scalar would work for the FD
        check. This form gives a closed-form analytic gradient (df/dphys =
        2*weights*phys) so the test can compute the chain-rule input directly
        without numerical noise.
        """
        weights = np.array([0.7, 1.3, -0.5, 0.9])[:, None]
        return float(np.sum(weights * phys**2))

    @staticmethod
    def _solver_imarr(seed=0, n=8, v_pre=None, m_pre=None, chi_pre=None):
        """Build a (4, n) solver-space image array.

        Rows are (imarr[0..3]) = (log_I, mprime, chi, vfrac_or_vprime); the
        meaning of slots 1 and 3 depends on which transforms are active.
        Passing a scalar for any of v_pre/m_pre/chi_pre pins that whole row
        — used to construct the diagonal-Jacobian operating points (mcv with
        vfrac=0, vcv with mfrac=0).
        """
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
        """transform_gradients(grad_phys, ...) ≈ FD of (f o transform_imarr).

        For each parametrized scenario:
          1. Build a solver-space imarr in the requested operating point.
          2. Compute phys = transform_imarr(imarr) and the analytic
             grad_phys = df/dphys = 2*weights*phys (the gradient of
             `_scalar_obj` w.r.t. its physical argument).
          3. Apply the chain rule under test:
                grad_solver = transform_gradients(grad_phys, imarr, ...)
          4. Compute the same df/dimarr numerically:
                grad_fd = centered FD of (f o transform_imarr)
          5. Assert grad_solver[k] ≈ grad_fd[k] for each solved-for row k.

        check_rows lists the rows to compare; non-solved rows have their
        gradient discarded by pack_imarr in real usage, so the value is
        irrelevant and not asserted.
        """
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
        """Direct check on the I-segment chain rule under log+mcv.

        With gradarr = ones, transform_gradients should produce
            outarr[0] = exp(imarr[0]) * gradarr[0] = exp(imarr[0])
        because the `log` block multiplies by exp(imarr[0]) and the mcv
        chain rule (which writes to slots 1:3) must not touch slot 0.
        """
        imarr = self._solver_imarr(seed=1, v_pre=0.0)
        gradarr = np.ones_like(imarr)
        which_solve = np.array([1, 1, 0, 0])

        out = transform_gradients(gradarr, imarr, ["log", "mcv"], which_solve)
        np.testing.assert_allclose(out[0], np.exp(imarr[0]), rtol=1e-12)

    def test_single_pol_log_chain_1d(self):
        """nimage==1 dispatch: 1D imarr (single-freq Stokes I, no pol).

        With `log`: outarr = exp(imarr) * gradarr (element-wise).
        Without `log`: outarr = gradarr (identity).
        """
        rng = np.random.default_rng(42)
        imarr = rng.uniform(-1.0, 1.0, size=8)
        gradarr = rng.uniform(-1.0, 1.0, size=8)
        which_solve = np.array([1, 0, 0, 0])

        out = transform_gradients(gradarr, imarr, ["log"], which_solve)
        np.testing.assert_allclose(out, np.exp(imarr) * gradarr, rtol=1e-12)

        out_no_log = transform_gradients(gradarr, imarr, [], which_solve)
        np.testing.assert_allclose(out_no_log, gradarr, rtol=1e-12)

    def test_multifreq_stokes_i_log_chain(self):
        """nimage==3 dispatch: (3, N) imarr (multifreq Stokes I, no pol).

        Layout is (log_I, alpha, beta) at the reference frequency. The `log`
        transform applies only to row 0 (Stokes I); the spectral coefficients
        in rows 1, 2 pass through unchanged.
        """
        rng = np.random.default_rng(7)
        imarr = rng.uniform(-1.0, 1.0, size=(3, 6))
        gradarr = rng.uniform(-1.0, 1.0, size=(3, 6))
        which_solve = np.array([1, 1, 1, 0])

        out = transform_gradients(gradarr, imarr, ["log"], which_solve)
        np.testing.assert_allclose(out[0], np.exp(imarr[0]) * gradarr[0], rtol=1e-12)
        np.testing.assert_array_equal(out[1], gradarr[1])
        np.testing.assert_array_equal(out[2], gradarr[2])

    def test_mf_pol_shape_smoke(self):
        """nimage==10 dispatch: (10, N) imarr (multifreq + full polarization).

        Layout is (I, mprime, chi, vfrac, alpha, beta, alphap, betap, rm, cm).
        Verifies the dispatcher (a) preserves shape, (b) applies `log` to slot
        0, and (c) leaves slots 4-9 (spectral / Faraday / conversion-measure
        coefficients) unchanged.
        """
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


class TestEmbedPackImarr:
    """Direct unit tests for embed_imarr + pack_imarr.

    embed_imarr extends a masked image array (nsolve, sum(mask)) into a full
    grid (nsolve, len(mask)) by inserting `clipfloor` (constant or
    `clipfloor * |N(0,1)|` if randomfloor=True) at non-mask pixels.

    pack_imarr stacks the rows of (nsolve, npix) that have which_solve[k] != 0
    into a single 1D solver vector. It does not consult the mask -- packing
    selects across the Stokes/spectral axis, not pixels.

    These two functions are not strict inverses of each other; their relation
    to unpack_imarr is covered in TestUnpackImarr.
    """

    @staticmethod
    def _make_mask(npix_total, npix_in, seed=0):
        """Build a boolean mask of length npix_total with exactly npix_in True."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(npix_total, size=npix_in, replace=False)
        mask = np.zeros(npix_total, dtype=bool)
        mask[idx] = True
        return mask

    # ------------------------------ embed_imarr ------------------------------

    def test_embed_1d_basic(self):
        """1D input dispatches through nsolve=1 branch and returns 1D output."""
        mask = self._make_mask(20, 7, seed=1)
        imvec = np.linspace(1.0, 7.0, 7)
        out = embed_imarr(imvec, mask, clipfloor=0.0)
        assert out.shape == (20,)
        np.testing.assert_array_equal(out[mask], imvec)
        np.testing.assert_array_equal(out[~mask], 0.0)

    def test_embed_2d_basic(self):
        """2D input (nsolve, npix_in) returns 2D output (nsolve, len(mask))."""
        mask = self._make_mask(16, 5, seed=2)
        imarr = np.arange(15, dtype=float).reshape(3, 5)
        out = embed_imarr(imarr, mask, clipfloor=0.0)
        assert out.shape == (3, 16)
        np.testing.assert_array_equal(out[:, mask], imarr)
        np.testing.assert_array_equal(out[:, ~mask], 0.0)

    def test_embed_clipfloor_constant(self):
        """Non-mask pixels get the constant clipfloor value when randomfloor=False."""
        mask = self._make_mask(12, 4, seed=3)
        imarr = np.array([[1.0, 2.0, 3.0, 4.0]])
        out = embed_imarr(imarr, mask, clipfloor=0.7, randomfloor=False)
        np.testing.assert_array_equal(out[0, ~mask], 0.7)
        np.testing.assert_array_equal(out[0, mask], imarr[0])

    def test_embed_randomfloor_seeded(self):
        """randomfloor=True with a fixed RNG seed is reproducible."""
        mask = self._make_mask(40, 10, seed=4)
        imvec = np.full(10, 5.0)

        np.random.seed(123)
        out1 = embed_imarr(imvec, mask, clipfloor=0.1, randomfloor=True)
        np.random.seed(123)
        out2 = embed_imarr(imvec, mask, clipfloor=0.1, randomfloor=True)
        np.testing.assert_array_equal(out1, out2)

        # Off-mask values are positive (|N(0,1)| scaled by clipfloor) and not constant.
        assert np.all(out1[~mask] >= 0.0)
        assert np.unique(out1[~mask]).size > 1

    def test_embed_all_true_mask_is_identity(self):
        """Mask fully True -> embed equals input (all pixels in mask, no fill needed)."""
        mask = np.ones(8, dtype=bool)
        imvec = np.linspace(-1.0, 1.0, 8)
        out = embed_imarr(imvec, mask, clipfloor=99.0)
        np.testing.assert_array_equal(out, imvec)

    def test_embed_wrong_imarr_dim_raises(self):
        """3D imarr is rejected; only 1D and 2D are supported."""
        mask = self._make_mask(10, 4, seed=5)
        bad = np.zeros((2, 2, 4))
        with pytest.raises(Exception, match="should have one or two dimensions"):
            embed_imarr(bad, mask)

    def test_embed_shape_mismatch_raises(self):
        """imarr's npix axis must equal sum(mask)."""
        mask = self._make_mask(10, 4, seed=6)
        bad = np.zeros((2, 3))  # second axis = 3, but sum(mask) = 4
        with pytest.raises(Exception, match="not consistent with imarr shape"):
            embed_imarr(bad, mask)

    # ------------------------------ pack_imarr -------------------------------

    def test_pack_stokes_i_only(self):
        """which_solve=[1,0,0,0] packs only row 0."""
        imarr = np.arange(40, dtype=float).reshape(4, 10)
        which_solve = np.array([1, 0, 0, 0])
        vec = pack_imarr(imarr, which_solve)
        assert vec.shape == (10,)
        np.testing.assert_array_equal(vec, imarr[0])

    def test_pack_full_iquv(self):
        """which_solve=[1,1,1,1] packs all four rows concatenated in order."""
        imarr = np.arange(20, dtype=float).reshape(4, 5)
        which_solve = np.array([1, 1, 1, 1])
        vec = pack_imarr(imarr, which_solve)
        assert vec.shape == (20,)
        np.testing.assert_array_equal(vec, imarr.reshape(-1))

    def test_pack_mixed_solved(self):
        """which_solve=[1,1,0,0] packs rows 0 and 1 in that order; 2 and 3 are skipped."""
        imarr = np.arange(20, dtype=float).reshape(4, 5)
        which_solve = np.array([1, 1, 0, 0])
        vec = pack_imarr(imarr, which_solve)
        np.testing.assert_array_equal(vec, np.hstack([imarr[0], imarr[1]]))

    def test_pack_multifreq_pol_spectral(self):
        """nimage=10 layout (mf+pol): which_solve picks I, mprime, alpha, beta."""
        imarr = np.arange(60, dtype=float).reshape(10, 6)
        which_solve = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        vec = pack_imarr(imarr, which_solve)
        expected = np.hstack([imarr[0], imarr[1], imarr[4], imarr[5]])
        np.testing.assert_array_equal(vec, expected)

    def test_pack_1d_input(self):
        """1D imarr with which_solve=[1] returns the input unchanged."""
        imarr = np.linspace(0.0, 1.0, 7)
        vec = pack_imarr(imarr, np.array([1]))
        np.testing.assert_array_equal(vec, imarr)

    def test_pack_no_solved_slots_returns_empty(self):
        """which_solve all zero yields an empty 1D array."""
        imarr = np.arange(12, dtype=float).reshape(3, 4)
        vec = pack_imarr(imarr, np.array([0, 0, 0]))
        assert vec.shape == (0,)

    def test_pack_which_solve_length_mismatch_raises(self):
        """which_solve length must equal imarr's nsolve dimension."""
        imarr = np.zeros((4, 5))
        with pytest.raises(Exception, match="inconsistent shape with which_solve"):
            pack_imarr(imarr, np.array([1, 0, 0]))

    def test_pack_wrong_dim_raises(self):
        """3D imarr is rejected; only 1D and 2D are supported."""
        with pytest.raises(Exception, match="should have one or two dimensions"):
            pack_imarr(np.zeros((2, 2, 4)), np.array([1, 1]))


class TestUnpackImarr:
    """Direct unit tests for unpack_imarr.

    unpack_imarr is the inverse of pack_imarr: it walks the rows of init_arr
    and, for each row k, either consumes the next nimage values from vec
    (when which_solve[k] != 0) or falls back to init_arr[k] (when 0).

    The pack -> unpack round-trip is the contract checked here. It is
    identity on solved-for rows; on unsolved rows it equals init_arr,
    not the original imarr the vec was packed from.
    """

    def test_round_trip_stokes_i(self):
        """Single-freq Stokes-I round-trip with which_solve=[1]."""
        rng = np.random.default_rng(0)
        imarr = rng.uniform(0.1, 1.0, size=(1, 16))
        which_solve = np.array([1])
        vec = pack_imarr(imarr, which_solve)
        out = unpack_imarr(vec, imarr, which_solve)
        np.testing.assert_array_equal(out, imarr)

    def test_round_trip_full_iquv(self):
        """which_solve all ones: pack/unpack is identity for any init_arr."""
        rng = np.random.default_rng(1)
        imarr = rng.uniform(-1.0, 1.0, size=(4, 12))
        init_arr = rng.uniform(-5.0, 5.0, size=(4, 12))
        which_solve = np.array([1, 1, 1, 1])
        vec = pack_imarr(imarr, which_solve)
        out = unpack_imarr(vec, init_arr, which_solve)
        # All rows solved -> unsolved fallback never engaged; init_arr is ignored.
        np.testing.assert_array_equal(out, imarr)

    def test_round_trip_partial_solve_uses_init_arr(self):
        """Unsolved rows of the output equal init_arr; solved rows equal imarr."""
        rng = np.random.default_rng(2)
        imarr = rng.uniform(-1.0, 1.0, size=(4, 8))
        init_arr = rng.uniform(-5.0, 5.0, size=(4, 8))
        which_solve = np.array([1, 1, 0, 0])  # IP-style: solve I and mprime
        vec = pack_imarr(imarr, which_solve)
        out = unpack_imarr(vec, init_arr, which_solve)
        np.testing.assert_array_equal(out[0], imarr[0])
        np.testing.assert_array_equal(out[1], imarr[1])
        np.testing.assert_array_equal(out[2], init_arr[2])
        np.testing.assert_array_equal(out[3], init_arr[3])

    def test_round_trip_iv(self):
        """IV-style which_solve=[1,0,0,1] picks rows 0 and 3 from vec."""
        rng = np.random.default_rng(3)
        imarr = rng.uniform(-1.0, 1.0, size=(4, 10))
        init_arr = rng.uniform(-5.0, 5.0, size=(4, 10))
        which_solve = np.array([1, 0, 0, 1])
        vec = pack_imarr(imarr, which_solve)
        out = unpack_imarr(vec, init_arr, which_solve)
        np.testing.assert_array_equal(out[0], imarr[0])
        np.testing.assert_array_equal(out[1], init_arr[1])
        np.testing.assert_array_equal(out[2], init_arr[2])
        np.testing.assert_array_equal(out[3], imarr[3])

    def test_round_trip_multifreq_pol(self):
        """nimage=10 (mf+pol): solve I, mprime, alpha, beta."""
        rng = np.random.default_rng(4)
        imarr = rng.uniform(-1.0, 1.0, size=(10, 6))
        init_arr = rng.uniform(-5.0, 5.0, size=(10, 6))
        which_solve = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        vec = pack_imarr(imarr, which_solve)
        out = unpack_imarr(vec, init_arr, which_solve)
        for k in (0, 1, 4, 5):
            np.testing.assert_array_equal(out[k], imarr[k])
        for k in (2, 3, 6, 7, 8, 9):
            np.testing.assert_array_equal(out[k], init_arr[k])

    def test_all_zero_which_solve_returns_init_arr(self):
        """When nothing is solved for, vec is empty and output equals init_arr."""
        init_arr = np.arange(20, dtype=float).reshape(4, 5)
        which_solve = np.array([0, 0, 0, 0])
        out = unpack_imarr(np.array([]), init_arr, which_solve)
        np.testing.assert_array_equal(out, init_arr)

    def test_1d_round_trip_solved(self):
        """1D init_arr with which_solve=[1]: vec is unpacked back into 1D shape."""
        rng = np.random.default_rng(7)
        imarr = rng.uniform(0.1, 1.0, size=8)  # 1D
        which_solve = np.array([1])
        vec = pack_imarr(imarr, which_solve)
        out = unpack_imarr(vec, imarr, which_solve)
        assert out.shape == imarr.shape
        np.testing.assert_array_equal(out, imarr)

    def test_1d_unsolved_falls_back_to_full_init_arr(self):
        """1D init_arr with which_solve=[0] returns init_arr unchanged.

        Regression for an internal reshape that was assigning to the wrong
        local name -- the unsolved fallback was broadcasting init_arr[0]
        (a scalar) across the row instead of using the whole 1D init_arr.
        """
        init_arr = np.linspace(0.1, 1.0, 8)
        out = unpack_imarr(np.array([]), init_arr, np.array([0]))
        assert out.shape == init_arr.shape
        np.testing.assert_array_equal(out, init_arr)

    def test_which_solve_length_mismatch_raises(self):
        """which_solve length must equal init_arr's nsolve dimension."""
        init_arr = np.zeros((4, 5))
        with pytest.raises(Exception, match="inconsistent shape with which_solve"):
            unpack_imarr(np.zeros(5), init_arr, np.array([1, 0, 0]))


class TestTransformImarr:
    """Direct unit tests for transform_imarr + transform_imarr_inverse.

    transform_imarr maps solver-space variables to physical Stokes-like
    variables and transform_imarr_inverse reverses it. The four transforms
    in production are:
      * log         -- imarr[0] (Stokes I) <-> exp(.) / log(.)
      * mcv         -- (mfrac_prime, vfrac) <-> (rho, psi) with mfrac in [0, 1-|v|]
      * vcv         -- (mfrac, vfrac_prime) <-> (rho, psi) with vfrac in [-(1-|m|), 1-|m|]
      * polcv       -- (rho_prime, psi_prime) <-> (rho, psi) with rho in (0,1), psi in (-pi/2, pi/2)

    Tests cover (a) the identity / no-op case, (b) the log dispatch across
    nimage = 1, 3, 4, 10, (c) round-trip identity for each transform on
    valid operating points, (d) specific value checks at well-known points,
    and (e) the mutually-exclusive transform guards.
    """

    @staticmethod
    def _pol_solver_imarr(seed=0, n=8, mfrac_prime=None, vfrac=None,
                          rho=None, psi=None, log_I=None):
        """Build a (4, n) solver-space image array.

        Slot meanings depend on the active transform (mcv vs vcv vs polcv);
        the helper just sets row values in solver space. Passing a scalar
        pins the row, None makes the row a smooth ramp -- this keeps test
        data deterministic without needing a separate RNG fixture per case.
        """
        rng = np.random.default_rng(seed)
        out = np.empty((4, n))
        out[0] = log_I if log_I is not None else rng.uniform(-1.0, 1.0, size=n)
        out[1] = mfrac_prime if mfrac_prime is not None else rng.uniform(-2.0, 2.0, size=n)
        out[2] = rng.uniform(-0.5, 0.5, size=n)  # phi (passes through)
        out[3] = vfrac if vfrac is not None else rng.uniform(-0.3, 0.3, size=n)
        # If caller specified physical (rho, psi) instead of solver values,
        # use those directly; the test will run inverse first.
        if rho is not None:
            out[1] = rho
        if psi is not None:
            out[3] = psi
        return out

    # --------------------------- identity / log only ---------------------------

    def test_no_transform_is_identity(self):
        """Empty transforms list returns imarr unchanged."""
        imarr = np.arange(16, dtype=float).reshape(4, 4)
        out = transform_imarr(imarr, [], np.array([1, 1, 1, 1]))
        np.testing.assert_array_equal(out, imarr)

    def test_log_1d_dispatch(self):
        """nimage=1 (1D imarr): out = exp(in)."""
        rng = np.random.default_rng(11)
        imarr = rng.uniform(-1.0, 1.0, size=8)
        out = transform_imarr(imarr, ["log"], np.array([1, 0, 0, 0]))
        np.testing.assert_allclose(out, np.exp(imarr), rtol=1e-12)

    def test_log_multifreq_stokes_i(self):
        """nimage=3 (mf no-pol): only row 0 (Stokes I) is exponentiated."""
        rng = np.random.default_rng(12)
        imarr = rng.uniform(-1.0, 1.0, size=(3, 5))
        out = transform_imarr(imarr, ["log"], np.array([1, 1, 1, 0]))
        np.testing.assert_allclose(out[0], np.exp(imarr[0]), rtol=1e-12)
        np.testing.assert_array_equal(out[1], imarr[1])
        np.testing.assert_array_equal(out[2], imarr[2])

    def test_log_single_freq_pol(self):
        """nimage=4 (sf pol): row 0 only is exponentiated."""
        imarr = self._pol_solver_imarr(seed=13, vfrac=0.0)
        which_solve = np.array([1, 1, 1, 0])
        out = transform_imarr(imarr, ["log"], which_solve)
        np.testing.assert_allclose(out[0], np.exp(imarr[0]), rtol=1e-12)
        np.testing.assert_array_equal(out[1:], imarr[1:])

    def test_log_multifreq_pol(self):
        """nimage=10 (mf+pol): row 0 only is exponentiated; rows 4-9 untouched."""
        rng = np.random.default_rng(14)
        imarr = np.empty((10, 6))
        imarr[0] = rng.uniform(-1.0, 1.0, size=6)
        imarr[1:4] = rng.uniform(-0.3, 0.3, size=(3, 6))
        imarr[4:10] = rng.uniform(-0.3, 0.3, size=(6, 6))
        which_solve = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        out = transform_imarr(imarr, ["log"], which_solve)
        np.testing.assert_allclose(out[0], np.exp(imarr[0]), rtol=1e-12)
        np.testing.assert_array_equal(out[1:], imarr[1:])

    # --------------------------- round-trip identity ---------------------------

    @pytest.mark.parametrize(
        "transforms,which_solve,kwargs",
        [
            (["log"], np.array([1, 0, 0, 0]), {}),
            (["log", "mcv"], np.array([1, 1, 1, 0]), {"vfrac": 0.0}),
            (["log", "mcv"], np.array([1, 1, 0, 0]), {}),
            (["log", "vcv"], np.array([1, 0, 0, 1]), {"mfrac_prime": 0.0}),
            (["log", "polcv"], np.array([1, 1, 0, 1]), {}),
            (["mcv"], np.array([0, 1, 0, 0]), {"vfrac": 0.0}),
            (["vcv"], np.array([0, 0, 0, 1]), {"mfrac_prime": 0.0}),
            (["polcv"], np.array([0, 1, 0, 1]), {}),
        ],
        ids=[
            "log_only", "log_mcv_IP", "log_mcv_general",
            "log_vcv_IV", "log_polcv_IPV",
            "mcv_only", "vcv_only", "polcv_only",
        ],
    )
    def test_round_trip_identity(self, transforms, which_solve, kwargs):
        """inverse(forward(imarr)) == imarr on valid operating points."""
        imarr = self._pol_solver_imarr(seed=42, **kwargs)
        phys = transform_imarr(imarr, transforms, which_solve)
        back = transform_imarr_inverse(phys, transforms, which_solve)
        np.testing.assert_allclose(back, imarr, atol=1e-10, rtol=1e-10)

    def test_round_trip_log_1d(self):
        """log inverse on 1D imarr is the inverse of exp."""
        rng = np.random.default_rng(21)
        imarr = rng.uniform(-1.0, 1.0, size=8)  # solver values (log I)
        phys = transform_imarr(imarr, ["log"], np.array([1, 0, 0, 0]))
        back = transform_imarr_inverse(phys, ["log"], np.array([1, 0, 0, 0]))
        np.testing.assert_allclose(back, imarr, rtol=1e-12)

    # --------------------------- specific value checks -------------------------

    def test_mcv_at_zero(self):
        """mcv at (mfrac_prime=0, vfrac=0): rho=0.5, psi=0.

        With mfrac_max = 1 - |vfrac| = 1 and arctan(0) = 0:
            mfrac = 1 * (0.5 + 0/pi) = 0.5
            rho = sqrt(0.5^2 + 0^2) = 0.5
            psi = arcsin(0 / 0.5) = 0
        """
        imarr = self._pol_solver_imarr(seed=0, n=4,
                                       mfrac_prime=0.0, vfrac=0.0)
        phys = transform_imarr(imarr, ["mcv"], np.array([1, 1, 0, 0]))
        np.testing.assert_allclose(phys[1], 0.5, rtol=1e-12)
        np.testing.assert_allclose(phys[3], 0.0, atol=1e-12)

    def test_vcv_at_zero(self):
        """vcv at (mfrac=0, vfrac_prime=0): rho=0, psi=0.

        With vfrac_max = 1 - |mfrac| = 1 and arctan(0) = 0:
            vfrac = 2 * 1 * 0 / pi = 0
            rho = sqrt(0 + 0) = 0
            psi = arcsin(0/0) -- handled inside, expect rho=0.
        """
        imarr = self._pol_solver_imarr(seed=0, n=4,
                                       mfrac_prime=0.0, vfrac=0.0)
        # In vcv the slot meanings are (I, mfrac, phi, vfrac_prime).
        phys = transform_imarr(imarr, ["vcv"], np.array([0, 0, 0, 1]))
        np.testing.assert_allclose(phys[1], 0.0, atol=1e-12)

    def test_polcv_at_zero(self):
        """polcv at (rho_prime=0, psi_prime=0): rho=0.5, psi=0."""
        imarr = self._pol_solver_imarr(seed=0, n=4,
                                       mfrac_prime=0.0, vfrac=0.0)
        # polcv slot meanings: (I, rho_prime, phi, psi_prime).
        phys = transform_imarr(imarr, ["polcv"], np.array([1, 1, 0, 1]))
        np.testing.assert_allclose(phys[1], 0.5, rtol=1e-12)
        np.testing.assert_allclose(phys[3], 0.0, atol=1e-12)

    # --------------------------- guards ---------------------------------------

    def test_polcv_with_mcv_raises(self):
        """polcv cannot be combined with mcv."""
        imarr = self._pol_solver_imarr(seed=0)
        with pytest.raises(Exception, match="not compatible with 'polcv'"):
            transform_imarr(imarr, ["polcv", "mcv"], np.array([1, 1, 0, 1]))

    def test_polcv_with_vcv_raises(self):
        """polcv cannot be combined with vcv."""
        imarr = self._pol_solver_imarr(seed=0)
        with pytest.raises(Exception, match="not compatible with 'polcv'"):
            transform_imarr(imarr, ["polcv", "vcv"], np.array([1, 1, 0, 1]))

    def test_mcv_with_vcv_raises(self):
        """mcv and vcv cannot be combined."""
        imarr = self._pol_solver_imarr(seed=0)
        with pytest.raises(Exception, match="not compatible with each other"):
            transform_imarr(imarr, ["mcv", "vcv"], np.array([1, 1, 0, 1]))

    def test_inverse_guards_match_forward(self):
        """Same conflict checks apply on the inverse path."""
        imarr = self._pol_solver_imarr(seed=0)
        with pytest.raises(Exception, match="not compatible"):
            transform_imarr_inverse(imarr, ["mcv", "vcv"], np.array([1, 1, 0, 1]))

    def test_invalid_nimage_raises(self):
        """nimage not in (1, 3, 4, 10) is rejected."""
        bad = np.zeros((2, 5))  # nimage = 2 is not a valid layout
        with pytest.raises(Exception, match="either 1, 3, 4, or 10"):
            transform_imarr(bad, ["log"], np.array([1, 0, 0, 0]))


class TestMakeInitarr:
    """Direct unit tests for make_initarr.

    Builds the initial-image array passed to the solver: a 1D Stokes-I
    vector (single-frequency, single-polarization), a (4, npix) array
    (single-frequency, full pol), a (3, npix) array (multifreq Stokes I),
    or a (10, npix) array (multifreq + full pol).

    The polarimetric branch optionally re-initializes rho and phi/psi from
    `np.random.rand` when `randompol_lin` or `randompol_circ` is set;
    those tests seed `np.random.seed(...)` directly so the assertions are
    deterministic. JAX migration will replace this with a
    `jax.random.PRNGKey` arg.
    """

    def test_stokes_only_no_norm(self, gauss_im):
        """No pol, no mf, no norm: initarr is the masked Stokes-I vector."""
        mask = np.ones(gauss_im.imvec.size, dtype=bool)
        out = make_initarr(gauss_im, mask, norm_init=False)
        assert out.ndim == 1
        np.testing.assert_array_equal(out, gauss_im.imvec)

    def test_stokes_only_norm_init(self, gauss_im):
        """norm_init rescales the masked Stokes-I vector to sum exactly to `flux`."""
        mask = np.ones(gauss_im.imvec.size, dtype=bool)
        target_flux = 2.5
        out = make_initarr(gauss_im, mask, norm_init=True, flux=target_flux)
        np.testing.assert_allclose(out.sum(), target_flux, rtol=1e-12)

    def test_stokes_only_with_mask(self, gauss_im):
        """Mask selects a subset of pixels; output length matches the mask."""
        mask = np.zeros(gauss_im.imvec.size, dtype=bool)
        mask[: mask.size // 2] = True
        out = make_initarr(gauss_im, mask, norm_init=False)
        assert out.shape == (mask.sum(),)
        np.testing.assert_array_equal(out, gauss_im.imvec[mask])

    def test_pol_shape_and_rows(self, gauss_im_pol):
        """pol=True, mf=False: returns (4, npix) with (I, rho, phi, psi).

        Loaded gauss_im_pol has nonzero Q/U/V, so without random reinit
        rho and phi are populated from image content rather than RNG.
        """
        mask = np.ones(gauss_im_pol.imvec.size, dtype=bool)
        out = make_initarr(gauss_im_pol, mask, pol=True)
        assert out.shape == (4, mask.sum())
        # init_rho is sqrt(Q^2 + U^2 + V^2) / I; positive everywhere here.
        assert np.all(out[1] >= 0)
        # phi = arctan2(U, Q) in [-pi, pi].
        assert np.all((out[2] >= -np.pi) & (out[2] <= np.pi))

    def test_pol_randompol_lin_seeded(self, gauss_im_pol):
        """randompol_lin=True with seeded RNG is deterministic."""
        mask = np.ones(gauss_im_pol.imvec.size, dtype=bool)

        np.random.seed(42)
        out1 = make_initarr(gauss_im_pol, mask, pol=True, randompol_lin=True,
                            meanpol=0.2, sigmapol=1e-2)
        np.random.seed(42)
        out2 = make_initarr(gauss_im_pol, mask, pol=True, randompol_lin=True,
                            meanpol=0.2, sigmapol=1e-2)
        np.testing.assert_array_equal(out1, out2)

        # rho ~ meanpol * (1 + sigmapol * U[0,1]); within (meanpol, meanpol*(1+sigmapol)).
        rho = out1[1]
        assert np.all(rho >= 0.2)
        assert np.all(rho <= 0.2 * (1 + 1e-2))

    def test_pol_randompol_lin_different_seeds_differ(self, gauss_im_pol):
        """Different seeds yield different random pol initializations."""
        mask = np.ones(gauss_im_pol.imvec.size, dtype=bool)
        np.random.seed(1)
        out1 = make_initarr(gauss_im_pol, mask, pol=True, randompol_lin=True)
        np.random.seed(2)
        out2 = make_initarr(gauss_im_pol, mask, pol=True, randompol_lin=True)
        assert not np.array_equal(out1[1], out2[1])

    def test_pol_randompol_circ_seeded(self, gauss_im_pol):
        """randompol_circ writes rho and psi (slot 3), not phi (slot 2)."""
        mask = np.ones(gauss_im_pol.imvec.size, dtype=bool)
        np.random.seed(7)
        out = make_initarr(gauss_im_pol, mask, pol=True, randompol_circ=True,
                           meanpol=0.2, sigmapol=1e-2)
        rho = out[1]
        psi = out[3]
        assert np.all(rho >= 0.2)
        assert np.all(rho <= 0.2 * (1 + 1e-2))
        assert np.all(np.abs(psi) <= 1e-2)

    def test_mf_stokes_shape(self, gauss_im):
        """mf=True, pol=False: returns (3, npix) with (I, alpha, beta).

        The base gauss_im has no specvec / curvvec, so alpha and beta
        are filled with zeros.
        """
        mask = np.ones(gauss_im.imvec.size, dtype=bool)
        out = make_initarr(gauss_im, mask, mf=True)
        assert out.shape == (3, mask.sum())
        np.testing.assert_array_equal(out[1], 0.0)
        np.testing.assert_array_equal(out[2], 0.0)

    def test_mf_pol_shape(self, gauss_im_pol):
        """mf=True, pol=True: returns (10, npix) layout (I, rho, phi, psi,
        alpha, beta, alphap, betap, rm, cm)."""
        mask = np.ones(gauss_im_pol.imvec.size, dtype=bool)
        out = make_initarr(gauss_im_pol, mask, mf=True, pol=True)
        assert out.shape == (10, mask.sum())
        # Stokes I row matches the (norm_init=False default) raw imvec.
        np.testing.assert_array_equal(out[0], gauss_im_pol.imvec[mask])
        # Spectral coefficients default to zero on an image with no specvec.
        np.testing.assert_array_equal(out[4], 0.0)
        np.testing.assert_array_equal(out[5], 0.0)
        np.testing.assert_array_equal(out[8], 0.0)
        np.testing.assert_array_equal(out[9], 0.0)


class TestValidateParams:
    """Direct unit tests for validate_params.

    Pure validator extracted from Imager.check_params. Each test pins one
    of the 24 raise paths by mutating a single field of a known-valid
    baseline config. Exception messages are checked verbatim so any drift
    in wording is a test failure.

    Baseline: Stokes-I direct-DFT imaging with the basic data + reg terms
    that Imager() defaults to. Each test calls _call(**overrides).
    """

    @staticmethod
    def _baseline(prior, init=None):
        return dict(
            prior=prior,
            init=init if init is not None else prior,
            pol='I',
            transforms=['log'],
            dat_term_keys=['vis'],
            reg_term_keys=['simple'],
            ttype='direct',
            mf=False,
            mf_order=0,
            mf_order_pol=0,
            freq_list=[230e9],
        )

    @staticmethod
    def _call(**kwargs):
        return validate_params(**kwargs)

    # Sanity: the baseline does not raise.
    def test_baseline_ok(self, gauss_im):
        self._call(**self._baseline(gauss_im))

    # ---- image cross-checks ----

    def test_dim_mismatch_raises(self, gauss_im):
        bigger = gauss_im.regrid_image(gauss_im.fovx(), 64)
        with pytest.raises(Exception,
                           match="Initial image does not match dimensions of the prior image!"):
            self._call(**self._baseline(gauss_im, init=bigger))

    def test_freq_mismatch_raises(self, gauss_im):
        other = gauss_im.copy()
        other.rf = 345e9
        with pytest.raises(Exception,
                           match="Initial image does not have same frequency as prior image!"):
            self._call(**self._baseline(gauss_im, init=other))

    @staticmethod
    def _circ_image(gauss_im):
        """Build a polrep='circ' image with the same grid as gauss_im."""
        circ_arr = gauss_im.imvec.reshape(gauss_im.ydim, gauss_im.xdim)
        return eh.image.Image(
            circ_arr, gauss_im.psize, gauss_im.ra, gauss_im.dec,
            polrep='circ', pol_prim='RR', rf=gauss_im.rf,
        )

    def test_polrep_mismatch_raises(self, gauss_im):
        other = self._circ_image(gauss_im)
        with pytest.raises(Exception,
                           match="Initial image polrep does not match prior polrep!"):
            self._call(**self._baseline(gauss_im, init=other))

    # ---- polrep × pol consistency ----

    def test_circ_polrep_bad_pol_raises(self, gauss_im):
        circ = self._circ_image(gauss_im)
        cfg = self._baseline(circ)
        cfg['pol'] = 'I'  # invalid for circ
        with pytest.raises(Exception,
                           match="polrep is 'circ': pol_next must be 'RR' or 'LL'"):
            self._call(**cfg)

    def test_stokes_polrep_bad_pol_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['pol'] = 'XX'  # not a valid stokes pol_next
        with pytest.raises(Exception,
                           match="polrep is 'stokes': pol_next must be in"):
            self._call(**cfg)

    # ---- transform × pol consistency ----

    def test_log_with_stokes_q_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['pol'] = 'Q'
        cfg['transforms'] = ['log']
        with pytest.raises(Exception,
                           match="Cannot image Stokes Q, U, V with log image transformation!"):
            self._call(**cfg)

    def test_gs_or_simple_reg_with_stokes_v_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['pol'] = 'V'
        cfg['transforms'] = []  # avoid the log-on-V raise above
        cfg['reg_term_keys'] = ['simple']
        with pytest.raises(Exception,
                           match="'simple' and 'gs' regularizers do not work"):
            self._call(**cfg)

    # ---- ttype ----

    def test_invalid_ttype_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['ttype'] = 'bogus'
        with pytest.raises(Exception,
                           match=r"Possible ttype values are 'fast', 'direct','nfft'!"):
            self._call(**cfg)

    # ---- multifrequency ----

    def test_mf_single_freq_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['mf'] = True
        cfg['freq_list'] = [230e9]  # only one freq
        with pytest.raises(Exception,
                           match="at least two frequencies for multifrequency imaging"):
            self._call(**cfg)

    def test_mf_order_bad_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['mf'] = True
        cfg['mf_order'] = 3
        cfg['freq_list'] = [230e9, 345e9]
        with pytest.raises(Exception, match=r"mf_order must be in \[0,1,2\]"):
            self._call(**cfg)

    def test_mf_pol_unsupported_pol_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['mf'] = True
        cfg['pol'] = 'IP'  # not in ['P', 'QU'] for mf pol
        cfg['transforms'] = ['log', 'mcv']
        cfg['ttype'] = 'direct'
        cfg['freq_list'] = [230e9, 345e9]
        with pytest.raises(Exception,
                           match="only support pol_next=P for polarization multifrequency"):
            self._call(**cfg)

    def test_mf_order_pol_bad_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['mf'] = True
        cfg['pol'] = 'P'
        cfg['transforms'] = ['mcv']
        cfg['ttype'] = 'direct'
        cfg['dat_term_keys'] = ['pvis']
        cfg['reg_term_keys'] = ['msimple']
        cfg['mf_order_pol'] = 3
        cfg['freq_list'] = [230e9, 345e9]
        with pytest.raises(Exception, match=r"mf_order_pol must be in \[0,1,2\]"):
            self._call(**cfg)

    # ---- pol × transform consistency ----

    def test_p_without_mcv_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['pol'] = 'IP'
        cfg['transforms'] = ['log']
        cfg['dat_term_keys'] = ['pvis']
        cfg['reg_term_keys'] = ['msimple']
        with pytest.raises(Exception, match=r"IP imaging requires 'mcv' transform!"):
            self._call(**cfg)

    def test_p_with_vcv_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['pol'] = 'IP'
        cfg['transforms'] = ['log', 'mcv', 'vcv']
        cfg['dat_term_keys'] = ['pvis']
        cfg['reg_term_keys'] = ['msimple']
        with pytest.raises(Exception, match=r"Cannot do IP imaging with 'vcv' transform!"):
            self._call(**cfg)

    def test_p_with_polcv_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['pol'] = 'IP'
        cfg['transforms'] = ['log', 'mcv', 'polcv']
        cfg['dat_term_keys'] = ['pvis']
        cfg['reg_term_keys'] = ['msimple']
        with pytest.raises(Exception,
                           match=r"Cannot do IP imaging only with 'polcv' transform!"):
            self._call(**cfg)

    def test_v_without_vcv_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['pol'] = 'IV'
        cfg['transforms'] = ['log']
        cfg['dat_term_keys'] = ['vvis']
        cfg['reg_term_keys'] = ['msimple']
        with pytest.raises(Exception, match=r"IV imaging requires 'vcv' transform!"):
            self._call(**cfg)

    def test_v_with_mcv_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['pol'] = 'IV'
        cfg['transforms'] = ['log', 'vcv', 'mcv']
        cfg['dat_term_keys'] = ['vvis']
        cfg['reg_term_keys'] = ['msimple']
        with pytest.raises(Exception, match=r"Cannot do IV imaging only with 'mcv' transform!"):
            self._call(**cfg)

    def test_v_with_polcv_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['pol'] = 'IV'
        cfg['transforms'] = ['log', 'vcv', 'polcv']
        cfg['dat_term_keys'] = ['vvis']
        cfg['reg_term_keys'] = ['msimple']
        with pytest.raises(Exception,
                           match=r"Cannot do IV imaging only with 'polcv' transform!"):
            self._call(**cfg)

    def test_iquv_without_polcv_raises(self, gauss_im):
        # IPV is in POLARIZATION_MODES but rejected by the earlier stokes-pol
        # check (the valid-stokes list omits 'IPV'); the polcv guard is reached
        # only via 'IQUV' for stokes-polrep images.
        cfg = self._baseline(gauss_im)
        cfg['pol'] = 'IQUV'
        cfg['transforms'] = ['log']
        cfg['dat_term_keys'] = ['pvis']
        cfg['reg_term_keys'] = ['msimple']
        with pytest.raises(Exception,
                           match="Linear\\+Circular polarization imaging requires 'polcv' transform!"):
            self._call(**cfg)

    def test_pol_with_fast_ttype_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['pol'] = 'IP'
        cfg['transforms'] = ['log', 'mcv']
        cfg['ttype'] = 'fast'
        cfg['dat_term_keys'] = ['pvis']
        cfg['reg_term_keys'] = ['msimple']
        with pytest.raises(Exception,
                           match="FFT not yet implemented in polarimetric imaging"):
            self._call(**cfg)

    # ---- data / regularizer term presence + validity ----

    def test_missing_data_term_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['dat_term_keys'] = []
        with pytest.raises(Exception, match="Must have at least one data term!"):
            self._call(**cfg)

    def test_missing_reg_term_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['reg_term_keys'] = []
        with pytest.raises(Exception, match="Must have at least one regularizer term!"):
            self._call(**cfg)

    def test_invalid_data_term_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['dat_term_keys'] = ['not_a_term']
        with pytest.raises(Exception, match="Invalid data term: valid data terms are:"):
            self._call(**cfg)

    def test_invalid_reg_term_raises(self, gauss_im):
        cfg = self._baseline(gauss_im)
        cfg['reg_term_keys'] = ['not_a_reg']
        with pytest.raises(Exception, match="Invalid regularizer: valid regularizers are:"):
            self._call(**cfg)


class TestValidateLimits:
    """Direct unit tests for validate_limits.

    Pure validator. Returns a list of warning strings instead of printing.
    Per-observation conditions:
      * pixel too coarse for largest baseline,
      * FOV too small for smallest nonzero baseline,
      * specified total flux outside [0.8, 1.2] * max(|vis amp|) (only for
        pol in {'I', 'RR', 'LL'}).
    """

    @staticmethod
    def _grid_copy(im, psize=None, xdim=None, ydim=None):
        """Make a prior-shaped Image by overriding grid params on a copy."""
        out = im.copy()
        if psize is not None:
            out.psize = psize
        if xdim is not None:
            out.xdim = xdim
        if ydim is not None:
            out.ydim = ydim
        return out

    @staticmethod
    def _matched_flux(obs):
        """Pick a flux value that won't trigger the flux warnings."""
        return float(np.max(np.abs(obs.unpack('amp')['amp'])))

    # ---- uv-resolution warnings ----

    def test_no_warnings_when_grid_and_flux_match(self, gauss_im, obs_direct):
        flux = self._matched_flux(obs_direct)
        out = validate_limits(gauss_im, [obs_direct], 'I', flux, [])
        assert out == []

    def test_warns_when_pixel_too_coarse(self, gauss_im, obs_direct):
        uvdists = obs_direct.unpack('uvdist')['uvdist']
        maxbl = float(np.max(uvdists))
        coarse = self._grid_copy(gauss_im, psize=2.0 / maxbl)  # uvmax = maxbl/2
        flux = self._matched_flux(obs_direct)
        out = validate_limits(coarse, [obs_direct], 'I', flux, [])
        assert any("Pixel size is larger" in m for m in out)

    def test_warns_when_fov_too_small(self, gauss_im, obs_direct):
        uvdists = obs_direct.unpack('uvdist')['uvdist']
        minbl = float(np.max(uvdists[uvdists > 0]))
        tiny = self._grid_copy(gauss_im, xdim=1, ydim=1, psize=0.5 / minbl)
        flux = self._matched_flux(obs_direct)
        out = validate_limits(tiny, [obs_direct], 'I', flux, [])
        assert any("Field of View is smaller" in m for m in out)

    def test_multi_obs_appends_per_obs(self, gauss_im, obs_direct):
        uvdists = obs_direct.unpack('uvdist')['uvdist']
        maxbl = float(np.max(uvdists))
        coarse = self._grid_copy(gauss_im, psize=2.0 / maxbl)
        flux = self._matched_flux(obs_direct)
        out = validate_limits(coarse, [obs_direct, obs_direct], 'I', flux, [])
        pixel_warnings = [m for m in out if "Pixel size is larger" in m]
        assert len(pixel_warnings) == 2

    def test_empty_obslist_returns_empty(self, gauss_im):
        assert validate_limits(gauss_im, [], 'I', 1.0, []) == []

    # ---- flux warnings ----

    def test_warns_when_flux_too_high(self, gauss_im, obs_direct):
        maxamp = self._matched_flux(obs_direct)
        out = validate_limits(gauss_im, [obs_direct], 'I', 2.0 * maxamp, [])
        assert any("> 120%" in m for m in out)

    def test_warns_when_flux_too_low(self, gauss_im, obs_direct):
        maxamp = self._matched_flux(obs_direct)
        out = validate_limits(gauss_im, [obs_direct], 'I', 0.5 * maxamp, [])
        assert any("< 80%" in m for m in out)

    def test_flux_check_skipped_for_polarimetric_pol(self, gauss_im, obs_direct):
        """Flux warnings only fire for total-flux pols ('I', 'RR', 'LL')."""
        maxamp = self._matched_flux(obs_direct)
        out = validate_limits(gauss_im, [obs_direct], 'IP', 2.0 * maxamp, [])
        assert all(("> 120%" not in m) and ("< 80%" not in m) for m in out)

    def test_mf_flux_picks_per_obs_value(self, gauss_im, obs_direct):
        """When len(mf_flux) == len(obslist), each obs uses its own flux."""
        maxamp = self._matched_flux(obs_direct)
        # First obs flux is good, second is too high.
        mf_flux = [maxamp, 2.0 * maxamp]
        out = validate_limits(gauss_im, [obs_direct, obs_direct], 'I',
                              maxamp, mf_flux)
        flux_warnings = [m for m in out if "> 120%" in m]
        assert len(flux_warnings) == 1  # only the second obs trips


class TestPolChisqdataKwargs:
    """Pol direct chisqdata leaves accept and ignore standard-chisqdata kwargs.

    Prerequisite for the unified compute_chisqdata_term dispatcher: pol leaves
    must tolerate the kwargs that standard leaves consume (pol, snrcut, debias,
    weighting, maxset, systematic_noise, systematic_cphase_noise, cp_uv_min)
    without choking. Pol data preparation does not use any of these knobs --
    they are simply absorbed by **kwargs.
    """

    @staticmethod
    def _full_mask(im):
        return np.ones(im.imvec.size, dtype=bool)

    @staticmethod
    def _kitchen_sink_kwargs():
        return dict(
            pol='IP',
            snrcut=3.0,
            debias=True,
            weighting='uniform',
            maxset=True,
            systematic_noise=0.05,
            systematic_cphase_noise=0.02,
            cp_uv_min=1e6,
        )

    def test_chisqdata_pvis_ignores_kwargs(self, gauss_im_pol, obs_direct):
        from ehtim.imaging.pol_imager_utils import chisqdata_pvis
        mask = self._full_mask(gauss_im_pol)
        base = chisqdata_pvis(obs_direct, gauss_im_pol, mask)
        ext = chisqdata_pvis(obs_direct, gauss_im_pol, mask, **self._kitchen_sink_kwargs())
        np.testing.assert_array_equal(base[0], ext[0])
        np.testing.assert_array_equal(base[1], ext[1])
        np.testing.assert_array_equal(base[2], ext[2])

    def test_chisqdata_m_ignores_kwargs(self, gauss_im_pol, obs_direct):
        from ehtim.imaging.pol_imager_utils import chisqdata_m
        mask = self._full_mask(gauss_im_pol)
        base = chisqdata_m(obs_direct, gauss_im_pol, mask)
        ext = chisqdata_m(obs_direct, gauss_im_pol, mask, **self._kitchen_sink_kwargs())
        np.testing.assert_array_equal(base[0], ext[0])
        np.testing.assert_array_equal(base[1], ext[1])
        np.testing.assert_array_equal(base[2], ext[2])

    def test_chisqdata_vvis_ignores_kwargs(self, gauss_im_pol, obs_direct):
        from ehtim.imaging.pol_imager_utils import chisqdata_vvis
        mask = self._full_mask(gauss_im_pol)
        base = chisqdata_vvis(obs_direct, gauss_im_pol, mask)
        ext = chisqdata_vvis(obs_direct, gauss_im_pol, mask, **self._kitchen_sink_kwargs())
        np.testing.assert_array_equal(base[0], ext[0])
        np.testing.assert_array_equal(base[1], ext[1])
        np.testing.assert_array_equal(base[2], ext[2])

    def test_chisqdata_pvis_nfft_still_accepts_fft_kwargs(self, gauss_im_pol, obs_nfft):
        from ehtim.imaging.pol_imager_utils import chisqdata_pvis_nfft
        mask = self._full_mask(gauss_im_pol)
        # Standard kwargs are inert; FFT kwargs (fft_pad_factor, p_rad) still work.
        out = chisqdata_pvis_nfft(obs_nfft, gauss_im_pol, mask,
                                  pol='IP', snrcut=3.0, fft_pad_factor=2, p_rad=2)
        assert len(out) == 3  # (vis, sigma, A) triple

    def test_chisqdata_m_nfft_still_accepts_fft_kwargs(self, gauss_im_pol, obs_nfft):
        from ehtim.imaging.pol_imager_utils import chisqdata_m_nfft
        mask = self._full_mask(gauss_im_pol)
        out = chisqdata_m_nfft(obs_nfft, gauss_im_pol, mask,
                               debias=True, fft_pad_factor=2, p_rad=2)
        assert len(out) == 3

    def test_chisqdata_vvis_nfft_still_accepts_fft_kwargs(self, gauss_im_pol, obs_nfft):
        from ehtim.imaging.pol_imager_utils import chisqdata_vvis_nfft
        mask = self._full_mask(gauss_im_pol)
        out = chisqdata_vvis_nfft(obs_nfft, gauss_im_pol, mask,
                                  weighting='uniform', fft_pad_factor=2, p_rad=2)
        assert len(out) == 3


class TestComputeChisqdataTerm:
    """Bit-identical parity tests for compute_chisqdata_term vs legacy dispatchers.

    The new dispatcher must produce exactly the same (data, sigma, A) triples
    as imutils.chisqdata + polutils.polchisqdata so callers (compute_data_tuples)
    can be rewired without behavior change.
    """

    STD_DTYPES = ['vis', 'amp', 'logamp', 'bs', 'cphase', 'cphase_diag',
                  'camp', 'logcamp', 'logcamp_diag']
    POL_DTYPES = ['pvis', 'm', 'vvis']

    @staticmethod
    def _kwargs():
        """Kwargs matching compute_data_tuples' current call into chisqdata."""
        return dict(
            maxset=False, debias=False, snrcut=0., weighting='natural',
            systematic_noise=0., systematic_cphase_noise=0., cp_uv_min=0.,
            order=3, fft_pad_factor=2, conv_func='gaussian', p_rad=2,
        )

    @staticmethod
    def _full_mask(im):
        return np.ones(im.imvec.size, dtype=bool)

    @classmethod
    def _equal_recursive(cls, a, b, path=""):
        """Recursive equality that handles dtype=object ragged arrays.

        `_diag` chisqdata returns data + sigma as dtype=object arrays whose
        elements are themselves variable-length numpy arrays (PR #233), and
        a future leaf could nest object-arrays inside object-arrays.
        assert_array_equal trips on element-wise comparison of object dtypes;
        this walks the structure recursively. Both inputs are checked for
        the object-dtype property to catch routing bugs that would change
        the return shape.
        """
        a_is_obj = isinstance(a, np.ndarray) and a.dtype == object
        b_is_obj = isinstance(b, np.ndarray) and b.dtype == object
        assert a_is_obj == b_is_obj, (
            f"at {path or '<root>'}: object-dtype mismatch "
            f"(a object={a_is_obj}, b object={b_is_obj})"
        )
        if a_is_obj:
            assert len(a) == len(b), (
                f"at {path or '<root>'}: length mismatch ({len(a)} vs {len(b)})"
            )
            for i, (aa, bb) in enumerate(zip(a, b)):
                cls._equal_recursive(aa, bb, path=f"{path}[{i}]")
        else:
            np.testing.assert_array_equal(a, b)

    @classmethod
    def _data_sigma_equal(cls, a, b):
        """Compare data + sigma of the (data, sigma, A) triple.

        A is intentionally not deep-compared: both new and legacy dispatchers
        call the same leaf helper, so the returned A is identical by
        construction. Deep-comparing A is fragile because some leaves return
        dtype=object ragged arrays and NFFTInfo wrappers that don't satisfy
        np.testing.assert_array_equal.
        """
        cls._equal_recursive(a[0], b[0])
        cls._equal_recursive(a[1], b[1])

    @pytest.mark.parametrize("dtype", STD_DTYPES)
    def test_parity_direct_standard(self, gauss_im, obs_direct, dtype):
        from ehtim.imaging.imager_utils import chisqdata
        mask = self._full_mask(gauss_im)
        kw = self._kwargs()
        legacy = chisqdata(obs_direct, gauss_im, mask, dtype, pol='I',
                           ttype='direct', **kw)
        new = compute_chisqdata_term(obs_direct, gauss_im, mask, dtype,
                                     ttype='direct', pol='I', **kw)
        self._data_sigma_equal(legacy, new)

    @pytest.mark.parametrize("dtype", STD_DTYPES)
    def test_parity_fast_standard(self, gauss_im, obs_fast, dtype):
        from ehtim.imaging.imager_utils import chisqdata
        mask = self._full_mask(gauss_im)
        kw = self._kwargs()
        legacy = chisqdata(obs_fast, gauss_im, mask, dtype, pol='I',
                           ttype='fast', **kw)
        new = compute_chisqdata_term(obs_fast, gauss_im, mask, dtype,
                                     ttype='fast', pol='I', **kw)
        self._data_sigma_equal(legacy, new)

    @pytest.mark.parametrize("dtype", STD_DTYPES)
    def test_parity_nfft_standard(self, gauss_im, obs_nfft, dtype):
        from ehtim.imaging.imager_utils import chisqdata
        mask = self._full_mask(gauss_im)
        kw = self._kwargs()
        legacy = chisqdata(obs_nfft, gauss_im, mask, dtype, pol='I',
                           ttype='nfft', **kw)
        new = compute_chisqdata_term(obs_nfft, gauss_im, mask, dtype,
                                     ttype='nfft', pol='I', **kw)
        self._data_sigma_equal(legacy, new)

    @pytest.mark.parametrize("dtype", POL_DTYPES)
    def test_parity_direct_pol(self, gauss_im_pol, obs_direct, dtype):
        from ehtim.imaging.pol_imager_utils import polchisqdata
        mask = self._full_mask(gauss_im_pol)
        kw_fft = dict(fft_pad_factor=2, conv_func='gaussian', p_rad=2)
        legacy = polchisqdata(obs_direct, gauss_im_pol, mask, dtype,
                              ttype='direct', **kw_fft)
        new = compute_chisqdata_term(obs_direct, gauss_im_pol, mask, dtype,
                                     ttype='direct', pol='IP', **kw_fft)
        self._data_sigma_equal(legacy, new)

    @pytest.mark.parametrize("dtype", POL_DTYPES)
    def test_parity_nfft_pol(self, gauss_im_pol, obs_nfft, dtype):
        from ehtim.imaging.pol_imager_utils import polchisqdata
        mask = self._full_mask(gauss_im_pol)
        kw_fft = dict(fft_pad_factor=2, conv_func='gaussian', p_rad=2)
        legacy = polchisqdata(obs_nfft, gauss_im_pol, mask, dtype,
                              ttype='nfft', **kw_fft)
        new = compute_chisqdata_term(obs_nfft, gauss_im_pol, mask, dtype,
                                     ttype='nfft', pol='IP', **kw_fft)
        self._data_sigma_equal(legacy, new)

    def test_unknown_dtype_raises(self, gauss_im, obs_direct):
        mask = self._full_mask(gauss_im)
        with pytest.raises(Exception, match="data term .* not recognized"):
            compute_chisqdata_term(obs_direct, gauss_im, mask, 'bogus',
                                   ttype='direct', pol='I')

    def test_pol_with_fast_raises(self, gauss_im_pol, obs_direct):
        mask = self._full_mask(gauss_im_pol)
        with pytest.raises(Exception, match="not supported for dtype"):
            compute_chisqdata_term(obs_direct, gauss_im_pol, mask, 'pvis',
                                   ttype='fast', pol='IP')

    def test_invalid_ttype_raises(self, gauss_im, obs_direct):
        mask = self._full_mask(gauss_im)
        with pytest.raises(Exception, match="Possible ttype values"):
            compute_chisqdata_term(obs_direct, gauss_im, mask, 'vis',
                                   ttype='bogus', pol='I')

    def test_standard_dtype_in_pol_mode_uses_stokes_I(self, gauss_im_pol, obs_direct):
        """In pol='IP' mode, a standard dtype ('vis') reads Stokes-I data.

        Equivalent to legacy chisqdata(..., pol='I') after the imager had
        forced dterm_pol='I'. Compare with the new dispatcher passing pol='IP'.
        """
        from ehtim.imaging.imager_utils import chisqdata
        mask = self._full_mask(gauss_im_pol)
        kw = self._kwargs()
        legacy_I = chisqdata(obs_direct, gauss_im_pol, mask, 'vis',
                             pol='I', ttype='direct', **kw)
        new_IP = compute_chisqdata_term(obs_direct, gauss_im_pol, mask, 'vis',
                                        ttype='direct', pol='IP', **kw)
        self._data_sigma_equal(legacy_I, new_IP)

    def test_standard_dtype_with_no_stokes_I_pol_raises(self, gauss_im_pol, obs_direct):
        """pol='P' (linear pol only, no Stokes-I) rejects standard data terms."""
        mask = self._full_mask(gauss_im_pol)
        with pytest.raises(Exception, match="cannot use dterm vis with pol=P"):
            compute_chisqdata_term(obs_direct, gauss_im_pol, mask, 'vis',
                                   ttype='direct', pol='P', **self._kwargs())


class _ChisqTermFixtures:
    """Helpers shared by TestComputeChisqTerm + TestComputeChisqgradTerm."""

    STD_DTYPES = TestComputeChisqdataTerm.STD_DTYPES
    POL_DTYPES = TestComputeChisqdataTerm.POL_DTYPES

    @staticmethod
    def _full_mask(im):
        return np.ones(im.imvec.size, dtype=bool)

    @staticmethod
    def _partial_mask(im, frac=0.7):
        """Mask with the central `frac` of pixels True."""
        n = im.imvec.size
        m = np.zeros(n, dtype=bool)
        # Mask the brightest fraction (Gaussian center) so the leaf gets
        # nontrivial data on the masked subset.
        cutoff = np.quantile(im.imvec, 1.0 - frac)
        m[im.imvec >= cutoff] = True
        return m

    @staticmethod
    def _imcur_pol(gauss_im_pol):
        """Build a (4, npix) physical-space imcur from gauss_im_pol."""
        I = gauss_im_pol.imvec
        Q = 0.10 * I
        U = 0.05 * I
        V = 0.02 * I
        # Convert to (I, rho, phi, psi) physical-space layout, matching
        # what transform_imarr produces.
        rho = np.sqrt(Q**2 + U**2 + V**2) / I
        phi = np.arctan2(U, Q)
        psi = np.arcsin(V / np.sqrt(Q**2 + U**2 + V**2 + 1e-30))
        return np.array([I, rho, phi, psi])

    @staticmethod
    def _data_tuple(obs, prior, mask, dtype, ttype, pol):
        """Get (A, data, sigma) for this (dtype, ttype) via the unified data-tuple dispatcher."""
        data, sigma, A = compute_chisqdata_term(
            obs, prior, mask, dtype, ttype=ttype, pol=pol,
        )
        return A, data, sigma


class TestComputeChisqTerm(_ChisqTermFixtures):
    """Bit-identical parity vs imutils.chisq / polutils.polchisq."""

    @pytest.mark.parametrize("dtype", _ChisqTermFixtures.STD_DTYPES)
    def test_parity_direct_standard(self, gauss_im, obs_direct, dtype):
        from ehtim.imaging.imager_utils import chisq
        mask = self._full_mask(gauss_im)
        A, data, sigma = self._data_tuple(obs_direct, gauss_im, mask, dtype,
                                          'direct', 'I')
        imcur = np.array([gauss_im.imvec])  # (1, npix) Stokes-I only
        legacy = chisq(gauss_im.imvec, A, data, sigma, dtype, ttype='direct',
                       mask=mask)
        new = compute_chisq_term(imcur, dtype, A, data, sigma,
                                 ttype='direct', mask=mask)
        np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("dtype", _ChisqTermFixtures.STD_DTYPES)
    def test_parity_fast_standard(self, gauss_im, obs_fast, dtype):
        from ehtim.imaging.imager_utils import chisq
        mask = self._partial_mask(gauss_im)
        A, data, sigma = self._data_tuple(obs_fast, gauss_im, mask, dtype,
                                          'fast', 'I')
        # Solver-space imcur: only the masked subset.
        imvec = gauss_im.imvec[mask]
        imcur = np.array([imvec])
        legacy = chisq(imvec, A, data, sigma, dtype, ttype='fast', mask=mask)
        new = compute_chisq_term(imcur, dtype, A, data, sigma,
                                 ttype='fast', mask=mask)
        np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("dtype", _ChisqTermFixtures.STD_DTYPES)
    def test_parity_nfft_standard(self, gauss_im, obs_nfft, dtype):
        from ehtim.imaging.imager_utils import chisq
        mask = self._partial_mask(gauss_im)
        A, data, sigma = self._data_tuple(obs_nfft, gauss_im, mask, dtype,
                                          'nfft', 'I')
        imvec = gauss_im.imvec[mask]
        imcur = np.array([imvec])
        legacy = chisq(imvec, A, data, sigma, dtype, ttype='nfft', mask=mask)
        new = compute_chisq_term(imcur, dtype, A, data, sigma,
                                 ttype='nfft', mask=mask)
        np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("dtype", _ChisqTermFixtures.POL_DTYPES)
    def test_parity_direct_pol(self, gauss_im_pol, obs_direct, dtype):
        from ehtim.imaging.pol_imager_utils import polchisq
        mask = self._full_mask(gauss_im_pol)
        A, data, sigma = self._data_tuple(obs_direct, gauss_im_pol, mask, dtype,
                                          'direct', 'IP')
        imcur = self._imcur_pol(gauss_im_pol)
        legacy = polchisq(imcur, A, data, sigma, dtype, ttype='direct', mask=mask)
        new = compute_chisq_term(imcur, dtype, A, data, sigma,
                                 ttype='direct', mask=mask)
        np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("dtype", _ChisqTermFixtures.POL_DTYPES)
    def test_parity_nfft_pol(self, gauss_im_pol, obs_nfft, dtype):
        from ehtim.imaging.pol_imager_utils import polchisq
        mask = self._partial_mask(gauss_im_pol)
        A, data, sigma = self._data_tuple(obs_nfft, gauss_im_pol, mask, dtype,
                                          'nfft', 'IP')
        imcur_full = self._imcur_pol(gauss_im_pol)
        imcur = imcur_full[:, mask]  # solver-space sub-array
        legacy = polchisq(imcur, A, data, sigma, dtype, ttype='nfft', mask=mask)
        new = compute_chisq_term(imcur, dtype, A, data, sigma,
                                 ttype='nfft', mask=mask)
        np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-15)

    def test_unknown_dtype_raises(self, gauss_im):
        imcur = np.array([gauss_im.imvec])
        with pytest.raises(Exception, match="data term .* not recognized"):
            compute_chisq_term(imcur, 'bogus', None, None, None, ttype='direct')

    def test_pol_with_fast_raises(self, gauss_im_pol):
        imcur = self._imcur_pol(gauss_im_pol)
        with pytest.raises(Exception, match="not supported for dtype"):
            compute_chisq_term(imcur, 'pvis', None, None, None, ttype='fast')

    def test_invalid_ttype_raises(self, gauss_im):
        imcur = np.array([gauss_im.imvec])
        with pytest.raises(Exception, match="Possible ttype values"):
            compute_chisq_term(imcur, 'vis', None, None, None, ttype='bogus')

    def test_1d_imcur_standard_dtype(self, gauss_im, obs_direct):
        """Non-pol mode passes a 1D imcur; standard dtype consumes it directly."""
        from ehtim.imaging.imager_utils import chisq
        mask = self._full_mask(gauss_im)
        A, data, sigma = self._data_tuple(obs_direct, gauss_im, mask, 'vis',
                                          'direct', 'I')
        imcur_1d = gauss_im.imvec  # 1D, non-pol mode
        legacy = chisq(imcur_1d, A, data, sigma, 'vis', ttype='direct', mask=mask)
        new = compute_chisq_term(imcur_1d, 'vis', A, data, sigma,
                                 ttype='direct', mask=mask)
        np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-15)


class TestComputeChisqgradTerm(_ChisqTermFixtures):
    """Bit-identical parity vs imutils.chisqgrad / polutils.polchisqgrad."""

    @pytest.mark.parametrize("dtype", _ChisqTermFixtures.STD_DTYPES)
    def test_parity_direct_standard(self, gauss_im, obs_direct, dtype):
        from ehtim.imaging.imager_utils import chisqgrad
        mask = self._full_mask(gauss_im)
        A, data, sigma = self._data_tuple(obs_direct, gauss_im, mask, dtype,
                                          'direct', 'I')
        imcur = np.array([gauss_im.imvec])
        legacy = chisqgrad(gauss_im.imvec, A, data, sigma, dtype,
                           ttype='direct', mask=mask)
        new = compute_chisqgrad_term(imcur, dtype, A, data, sigma,
                                     ttype='direct', mask=mask)
        np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("dtype", _ChisqTermFixtures.STD_DTYPES)
    def test_parity_fast_standard(self, gauss_im, obs_fast, dtype):
        from ehtim.imaging.imager_utils import chisqgrad
        mask = self._partial_mask(gauss_im)
        A, data, sigma = self._data_tuple(obs_fast, gauss_im, mask, dtype,
                                          'fast', 'I')
        imvec = gauss_im.imvec[mask]
        imcur = np.array([imvec])
        legacy = chisqgrad(imvec, A, data, sigma, dtype, ttype='fast', mask=mask)
        new = compute_chisqgrad_term(imcur, dtype, A, data, sigma,
                                     ttype='fast', mask=mask)
        np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("dtype", _ChisqTermFixtures.STD_DTYPES)
    def test_parity_nfft_standard(self, gauss_im, obs_nfft, dtype):
        from ehtim.imaging.imager_utils import chisqgrad
        mask = self._partial_mask(gauss_im)
        A, data, sigma = self._data_tuple(obs_nfft, gauss_im, mask, dtype,
                                          'nfft', 'I')
        imvec = gauss_im.imvec[mask]
        imcur = np.array([imvec])
        legacy = chisqgrad(imvec, A, data, sigma, dtype, ttype='nfft', mask=mask)
        new = compute_chisqgrad_term(imcur, dtype, A, data, sigma,
                                     ttype='nfft', mask=mask)
        np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("dtype", _ChisqTermFixtures.POL_DTYPES)
    def test_parity_direct_pol(self, gauss_im_pol, obs_direct, dtype):
        from ehtim.imaging.pol_imager_utils import polchisqgrad
        mask = self._full_mask(gauss_im_pol)
        A, data, sigma = self._data_tuple(obs_direct, gauss_im_pol, mask, dtype,
                                          'direct', 'IP')
        imcur = self._imcur_pol(gauss_im_pol)
        pol_solve = np.ones(4, dtype=int)
        legacy = polchisqgrad(imcur, A, data, sigma, dtype, ttype='direct',
                              mask=mask, pol_solve=pol_solve)
        new = compute_chisqgrad_term(imcur, dtype, A, data, sigma,
                                     ttype='direct', mask=mask,
                                     pol_solve=pol_solve)
        np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("dtype", _ChisqTermFixtures.POL_DTYPES)
    def test_parity_nfft_pol(self, gauss_im_pol, obs_nfft, dtype):
        from ehtim.imaging.pol_imager_utils import polchisqgrad
        mask = self._partial_mask(gauss_im_pol)
        A, data, sigma = self._data_tuple(obs_nfft, gauss_im_pol, mask, dtype,
                                          'nfft', 'IP')
        imcur_full = self._imcur_pol(gauss_im_pol)
        imcur = imcur_full[:, mask]
        pol_solve = np.ones(4, dtype=int)
        legacy = polchisqgrad(imcur, A, data, sigma, dtype, ttype='nfft',
                              mask=mask, pol_solve=pol_solve)
        new = compute_chisqgrad_term(imcur, dtype, A, data, sigma,
                                     ttype='nfft', mask=mask,
                                     pol_solve=pol_solve)
        np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-15)

    def test_pol_grad_missing_pol_solve_raises(self, gauss_im_pol, obs_direct):
        """Pol grad requires explicit pol_solve; no hidden default."""
        mask = self._full_mask(gauss_im_pol)
        A, data, sigma = self._data_tuple(obs_direct, gauss_im_pol, mask, 'pvis',
                                          'direct', 'IP')
        imcur = self._imcur_pol(gauss_im_pol)
        with pytest.raises(Exception, match="requires explicit pol_solve"):
            compute_chisqgrad_term(imcur, 'pvis', A, data, sigma,
                                   ttype='direct', mask=mask)


class TestPolSolveBlock:
    """Tests for _pol_solve_block.

    Slices a polarimetric Stokes block out of which_solve. The function is
    a seam for the future WhichSolve(stokes, spectral) NamedTuple refactor;
    today it's a static 4-wide slice for the multifrequency + pol case.
    """

    def test_singlefreq_stokes_i_passthrough(self):
        """Single-freq Stokes-I: 1-wide which_solve, non-pol mode -> identity."""
        ws = np.array([1])
        out = _pol_solve_block(ws, pol='I')
        np.testing.assert_array_equal(out, ws)

    def test_singlefreq_pol_passthrough(self):
        """Single-freq pol: 4-wide which_solve, pol mode -> identity (no slicing needed)."""
        ws = np.array([1, 1, 1, 0])
        out = _pol_solve_block(ws, pol='IP')
        np.testing.assert_array_equal(out, ws)

    def test_multifreq_stokes_i_passthrough(self):
        """Multifreq Stokes-I: 3-wide which_solve, non-pol mode -> identity."""
        ws = np.array([1, 1, 1])
        out = _pol_solve_block(ws, pol='I')
        np.testing.assert_array_equal(out, ws)

    def test_multifreq_pol_sliced(self):
        """Multifreq pol: 10-wide which_solve, pol mode -> first 4 entries."""
        ws = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        out = _pol_solve_block(ws, pol='IP')
        np.testing.assert_array_equal(out, ws[:4])

    def test_multifreq_pol_non_pol_mode_passthrough(self):
        """If pol mode is not in POLARIZATION_MODES, do not slice even if length > 4."""
        ws = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        out = _pol_solve_block(ws, pol='I')
        np.testing.assert_array_equal(out, ws)

    def test_pol_mode_short_which_solve_passthrough(self):
        """Pol mode but 4-wide which_solve: no slicing (single-frequency case)."""
        ws = np.array([1, 0, 0, 1])
        out = _pol_solve_block(ws, pol='IV')
        np.testing.assert_array_equal(out, ws)

