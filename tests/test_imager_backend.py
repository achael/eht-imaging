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


class TestComputeChisqDict:
    """Tests for compute_chisq_dict (extracted from Imager.make_chisq_dict)."""

    def test_stokes_i_single_term(self, gauss_im, observe, initialize_imager):
        """Stokes I, single data term — returns one finite chi^2."""
        obs = observe(gauss_im)
        imgr, imcur = initialize_imager(obs, gauss_im, {"vis": 100})
        result = _call_backend_chisq_dict(imgr, imcur)
        assert set(result.keys()) == {"vis"}
        assert np.isfinite(result["vis"])

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

    def test_stokes_i_single_term(self, gauss_im, observe, initialize_imager):
        """Stokes I gradient has shape (nimage,) and is finite."""
        obs = observe(gauss_im)
        imgr, imcur = initialize_imager(obs, gauss_im, {"vis": 100})
        result = _call_backend_chisqgrad_dict(imgr, imcur)
        assert set(result.keys()) == {"vis"}
        assert result["vis"].shape == (imgr._nimage,)
        assert np.all(np.isfinite(result["vis"]))

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

